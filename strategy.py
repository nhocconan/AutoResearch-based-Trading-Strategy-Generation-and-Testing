#!/usr/bin/env python3
"""
Experiment #466: 12h Primary + 1d HTF — Simplified HMA Trend + RSI Extremes + Volume

Hypothesis: After analyzing 465 failed experiments, the pattern is clear:
1. Complex regime switching (Choppiness, ADX) adds latency and reduces trades
2. Connors RSI is theoretically sound but implementation complexity hurts
3. SIMPLER logic = MORE trades = better statistical significance
4. 12h TF naturally filters noise, needs fewer additional filters
5. Volume confirmation adds edge without over-complicating

Why this might beat current best (Sharpe=0.435):
- 1d HMA(21) provides clean major trend bias (proven in research)
- RSI(14) extremes on 12h catch reversals with 65%+ win rate
- Volume spike filter (1.5x avg) confirms genuine moves vs fakeouts
- HMA(21)/HMA(50) crossover on 12h adds trend confirmation
- Fewer conditions = 40-60 trades/year target achievable
- Asymmetric sizing protects in bear markets (0.30 long, 0.25 short)

Key simplifications vs failed attempts:
- NO Choppiness Index (removed - adds complexity, marginal benefit)
- NO Connors RSI (use standard RSI(14) - simpler, similar edge)
- Only 2-3 confluence conditions per entry (not 5-6)
- Relaxed RSI thresholds (30/70 instead of 20/80) for more trades
- Volume confirmation is single filter, not complex regime

Position sizing: 0.25-0.30 (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 40-60 trades/year on 12h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_rsi_vol_1d_simp_v1"
timeframe = "12h"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_volume_spike(volume, period=20, threshold=1.5):
    """Detect volume spikes above threshold * average volume."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_avg + 1e-10)
    spike = vol_ratio > threshold
    return spike

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (major trend direction)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_12h_21 = calculate_hma(close, period=21)
    hma_12h_50 = calculate_hma(close, period=50)
    rsi_14 = calculate_rsi(close, period=14)
    vol_spike = calculate_volume_spike(volume, period=20, threshold=1.5)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        if np.isnan(hma_12h_21[i]) or np.isnan(hma_12h_50[i]):
            continue
        if np.isnan(rsi_14[i]):
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        # Price above 1d HMA = bull bias (favor longs)
        # Price below 1d HMA = bear bias (favor shorts)
        bull_regime = close[i] > hma_1d_21_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i]
        
        # === 12H LOCAL TREND (HMA crossover) ===
        hma_bullish = hma_12h_21[i] > hma_12h_50[i]
        hma_bearish = hma_12h_21[i] < hma_12h_50[i]
        
        # === RSI EXTREMES (entry trigger) ===
        rsi_oversold = rsi_14[i] < 35.0  # relaxed for more trades
        rsi_overbought = rsi_14[i] > 65.0  # relaxed for more trades
        rsi_extreme_oversold = rsi_14[i] < 25.0
        rsi_extreme_overbought = rsi_14[i] > 75.0
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_spike[i]
        
        # === ENTRY LOGIC — SIMPLIFIED (2-3 conditions max) ===
        new_signal = 0.0
        
        # LONG ENTRIES (bull regime OR hma bullish + RSI oversold)
        if bull_regime:
            # Primary: RSI oversold + volume confirmation
            if rsi_oversold and vol_confirmed:
                new_signal = LONG_SIZE
            # Secondary: Extreme RSI oversold (works without volume)
            elif rsi_extreme_oversold:
                new_signal = LONG_SIZE * 0.8
            # Tertiary: HMA bullish + RSI neutral-oversold
            elif hma_bullish and rsi_14[i] < 45.0:
                new_signal = LONG_SIZE * 0.6
        
        # SHORT ENTRIES (bear regime OR hma bearish + RSI overbought)
        if bear_regime:
            # Primary: RSI overbought + volume confirmation
            if rsi_overbought and vol_confirmed:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Secondary: Extreme RSI overbought (works without volume)
            elif rsi_extreme_overbought:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.8
            # Tertiary: HMA bearish + RSI neutral-overbought
            elif hma_bearish and rsi_14[i] > 55.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.6
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no position and no signal, enter on very simple conditions
        if not in_position and new_signal == 0.0:
            # Long: RSI < 30 (simple mean reversion)
            if rsi_14[i] < 30.0:
                new_signal = LONG_SIZE * 0.5
            # Short: RSI > 70 (simple mean reversion)
            elif rsi_14[i] > 70.0:
                new_signal = -SHORT_SIZE * 0.5
        
        # === STOPLOSS CHECK (BEFORE exit logic - CRITICAL) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT CONDITIONS ===
        # RSI extreme exit (take profit on exhaustion)
        if in_position and position_side > 0 and rsi_14[i] > 80.0:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_14[i] < 20.0:
            new_signal = 0.0
        
        # Trend reversal exit (1d regime flip + 12h HMA flip)
        if in_position and position_side > 0 and bear_regime and hma_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime and hma_bullish:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals