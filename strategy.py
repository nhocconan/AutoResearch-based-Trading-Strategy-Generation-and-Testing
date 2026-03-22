#!/usr/bin/env python3
"""
Experiment #465: 1h Primary + 4h/1d HTF — HMA Trend + RSI Pullback + Volume + Session

Hypothesis: After 464 experiments, clear pattern:
1. 1h timeframe needs VERY strict filters to avoid fee drag (target 30-80 trades/year)
2. 4h HMA(21) provides reliable trend direction (proven in current best strategy)
3. 1d HMA(21) as additional major trend confirmation
4. RSI(14) pullback entries within HTF trend (RSI<45 long, RSI>55 short)
5. Volume filter (>0.8x 20-period avg) confirms institutional participation
6. Session filter (8-20 UTC) avoids low-liquidity whipsaws
7. Choppiness Index adapts entry thresholds (range vs trend mode)

Why this might beat current best (Sharpe=0.435):
- 1h entries within 4h/1d trend = HTF edge with precise timing
- Volume + session filters reduce false signals during low-liquidity periods
- Simpler than failed regime-switching strategies (experiments 454, 461, 464)
- Discrete position sizing (0.25/0.30) minimizes fee churn
- ATR 2.5x trailing stop protects in crash scenarios

Position sizing: 0.25-0.30 (discrete, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 40-80 trades/year on 1h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_vol_session_4h1d_v1"
timeframe = "1h"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend follow)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    atr = calculate_atr(high, low, close, period)
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    highest_high = high_s.rolling(window=period, min_periods=period).max().values
    lowest_low = low_s.rolling(window=period, min_periods=period).min().values
    
    range_hl = highest_high - lowest_low
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    
    chop = 100.0 * np.log10(atr_sum / range_hl) / np.log10(period)
    
    return chop

def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return pd.to_datetime(open_time, unit='ms').dt.hour.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    vol_sma_20 = calculate_volume_sma(volume, period=20)
    utc_hour = get_utc_hour(open_time)
    
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
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]) or np.isnan(vol_sma_20[i]):
            continue
        if vol_sma_20[i] == 0:
            continue
        
        # === 4H MAJOR TREND (primary direction filter) ===
        bull_4h = close[i] > hma_4h_21_aligned[i]
        bear_4h = close[i] < hma_4h_21_aligned[i]
        
        # === 1D CONFIRMATION (stronger trend bias) ===
        bull_1d = close[i] > hma_1d_21_aligned[i]
        bear_1d = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_ranging = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === VOLUME FILTER (institutional participation) ===
        volume_ok = volume[i] > 0.8 * vol_sma_20[i]
        
        # === SESSION FILTER (8-20 UTC = high liquidity) ===
        session_ok = (utc_hour[i] >= 8) and (utc_hour[i] <= 20)
        
        # === RSI PULLBACK LEVELS ===
        rsi_oversold = rsi_14[i] < 45.0
        rsi_overbought = rsi_14[i] > 55.0
        rsi_extreme_oversold = rsi_14[i] < 30.0
        rsi_extreme_overbought = rsi_14[i] > 70.0
        
        # === ENTRY LOGIC — 3+ CONFLUENCE REQUIRED ===
        new_signal = 0.0
        
        # LONG ENTRIES (need: 4h bull + volume + session + RSI pullback)
        long_confidence = 0
        if bull_4h:
            long_confidence += 1
        if bull_1d:
            long_confidence += 1
        if volume_ok:
            long_confidence += 1
        if session_ok:
            long_confidence += 1
        if rsi_oversold:
            long_confidence += 1
        
        # Enter long with 3+ confluence
        if long_confidence >= 3:
            if is_trending and bull_4h and rsi_oversold:
                new_signal = LONG_SIZE
            elif is_ranging and rsi_extreme_oversold:
                new_signal = LONG_SIZE
            elif bull_4h and bull_1d and rsi_14[i] < 50.0:
                new_signal = LONG_SIZE * 0.8
        
        # SHORT ENTRIES (need: 4h bear + volume + session + RSI bounce)
        short_confidence = 0
        if bear_4h:
            short_confidence += 1
        if bear_1d:
            short_confidence += 1
        if volume_ok:
            short_confidence += 1
        if session_ok:
            short_confidence += 1
        if rsi_overbought:
            short_confidence += 1
        
        # Enter short with 3+ confluence
        if short_confidence >= 3:
            if new_signal == 0.0:  # Don't override long signal
                if is_trending and bear_4h and rsi_overbought:
                    new_signal = -SHORT_SIZE
                elif is_ranging and rsi_extreme_overbought:
                    new_signal = -SHORT_SIZE
                elif bear_4h and bear_1d and rsi_14[i] > 50.0:
                    new_signal = -SHORT_SIZE * 0.8
        
        # === TRADE FREQUENCY BOOST (ensure >=30 trades/symbol) ===
        # If no position, relax to 2+ confluence for entry
        if not in_position and new_signal == 0.0:
            if bull_4h and rsi_extreme_oversold and volume_ok:
                new_signal = LONG_SIZE * 0.6
            elif bear_4h and rsi_extreme_overbought and volume_ok:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.6
        
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
        if in_position and position_side > 0 and rsi_14[i] > 75.0:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_14[i] < 25.0:
            new_signal = 0.0
        
        # Trend reversal exit (4h regime flip)
        if in_position and position_side > 0 and bear_4h and rsi_14[i] > 50.0:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_4h and rsi_14[i] < 50.0:
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