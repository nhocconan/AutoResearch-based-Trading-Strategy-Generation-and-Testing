#!/usr/bin/env python3
"""
Experiment #518: 30m Primary + 4h/1d HTF — Simplified Regime-Adaptive Mean Reversion

Hypothesis: After 464 failed strategies (mostly over-filtered), try a SIMPLER approach
that generates TRADES while maintaining quality:

1. 4h HMA(21) = major trend direction (long when price > 4h HMA, short when <)
2. RSI(3) = fast mean reversion (RSI < 20 long, RSI > 80 short) - faster than RSI(14)
3. Choppiness Index(14) = regime awareness (not strict filter, just modifies thresholds)
4. Volume filter = volume > 0.5x 20-bar avg (light filter, avoid dead hours only)
5. NO session filter (crypto is 24/7, session filter killed trades in exp #508)

Why this might beat Sharpe=0.435:
- 30m with 4h direction = HTF trade frequency with 30m entry precision
- RSI(3) catches quick reversals that RSI(14) misses
- Simpler filters = MORE trades (critical: exp #508 got 0 trades with CRSI+CHOP+HMA)
- Looser thresholds ensure >=30 trades/symbol on train

Position sizing: 0.25 (discrete, max 0.40 for lower TF)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 40-60 trades/year on 30m, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_rsi3_hma4h_chop_v1"
timeframe = "30m"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Highest high and lowest low over period
    highest = high_s.rolling(window=period, min_periods=period).max()
    lowest = low_s.rolling(window=period, min_periods=period).min()
    
    # ATR sum over period
    atr_vals = calculate_atr(high, low, close, period)
    atr_sum = pd.Series(atr_vals).rolling(window=period, min_periods=period).sum()
    
    # CHOP formula
    range_hl = highest - lowest
    range_hl = range_hl.replace(0, 1e-10)
    
    chop = 100.0 * np.log10(atr_sum / range_hl) / np.log10(period)
    
    return chop.values

def calculate_rsi(close, period=3):
    """Calculate fast RSI for quick mean reversion signals."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for trend direction
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    rsi_3 = calculate_rsi(close, 3)
    
    # Volume average for filter
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_4h_aligned[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(rsi_3[i]):
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] == 0:
            continue
        
        # === 4H TREND DIRECTION ===
        bull_trend = close[i] > hma_4h_aligned[i]
        bear_trend = close[i] < hma_4h_aligned[i]
        
        # === CHOPPINESS REGIME ===
        range_regime = chop_14[i] > 50.0  # Choppy/ranging
        trend_regime = chop_14[i] <= 50.0  # Trending
        
        # === RSI MEAN REVERSION (fast RSI(3)) ===
        # Looser thresholds to ensure trades (exp #508 failed with too strict)
        rsi_oversold = rsi_3[i] < 20.0
        rsi_overbought = rsi_3[i] > 80.0
        
        # === VOLUME FILTER (light) ===
        vol_ok = volume[i] > 0.5 * vol_avg[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG: Bull trend + oversold RSI + volume ok
        if bull_trend and vol_ok:
            if rsi_oversold:
                new_signal = SIZE
            # Also enter in range regime with less extreme RSI
            elif range_regime and rsi_3[i] < 30.0:
                new_signal = SIZE * 0.8
        
        # SHORT: Bear trend + overbought RSI + volume ok
        if new_signal == 0.0 and bear_trend and vol_ok:
            if rsi_overbought:
                new_signal = -SIZE
            # Also enter in range regime with less extreme RSI
            elif range_regime and rsi_3[i] > 70.0:
                new_signal = -SIZE * 0.8
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON REGIME CHANGE ===
        if in_position and position_side > 0 and bear_trend:
            new_signal = 0.0
        
        if in_position and position_side < 0 and bull_trend:
            new_signal = 0.0
        
        # === EXIT ON RSI REVERSAL (take profit) ===
        if in_position and position_side > 0 and rsi_3[i] > 70.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and rsi_3[i] < 30.0:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals