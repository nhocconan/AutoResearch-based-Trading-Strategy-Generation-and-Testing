#!/usr/bin/env python3
"""
Experiment #142: 12h Primary + 1d/1w HTF — Simplified HMA + RSI + Regime

Hypothesis: Previous 12h strategies failed due to overly strict entry conditions
(0 trades). This version uses LOOSER thresholds that actually trigger:

1) 1d HMA(21) for macro trend bias — only trade with trend
2) 12h HMA(16/48) crossover for entry timing
3) RSI(14) filter: >45 for longs, <55 for shorts (NOT extreme values)
4) Choppiness Index(14): >55 = reduce size (choppy), <45 = full size (trending)
5) ADX(14) > 20 confirms trend strength
6) ATR(14) trailing stop at 2.5x
7) Exit: opposite HMA crossover or stoploss

Why this should work:
- Simpler than #136, #139 which had 0 trades or negative Sharpe
- RSI thresholds (45/55) are much easier to hit than (20/80)
- 12h naturally produces 25-40 trades/year (low fee drag)
- HMA crossover proven in trend markets
- Regime filter reduces size in chop, not blocking entries

Position size: 0.25 base, 0.30 with ADX confirmation
Stoploss: 2.5*ATR trailing
Target: 30-50 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_rsi_regime_1d1w_v1"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50).values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    plus_dm = plus_dm.clip(lower=0)
    minus_dm = minus_dm.clip(lower=0)
    
    # Smooth DM and TR
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100.0 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    
    # DX and ADX
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.fillna(0).values

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppy vs trending.
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.maximum(price_range, 1e-10)
    
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.nan_to_num(chop, nan=50.0)
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA for macro trend
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1w HMA for ultra-macro filter
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    rsi_14 = calculate_rsi(close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    POSITION_SIZE_REDUCED = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]) or np.isnan(atr_14[i]):
            continue
        if atr_14[i] == 0 or np.isnan(rsi_14[i]) or np.isnan(adx_14[i]):
            continue
        if np.isnan(chop_14[i]):
            continue
        
        # === HTF TREND BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        price_above_hma_1w = close[i] > hma_1w_aligned[i] if not np.isnan(hma_1w_aligned[i]) else True
        
        # === LOCAL TREND (HMA crossover) ===
        hma_bullish = hma_16[i] > hma_48[i]
        hma_bearish = hma_16[i] < hma_48[i]
        
        # Check for crossover (entry signal)
        hma_cross_long = hma_bullish and (hma_16[i-1] <= hma_48[i-1])
        hma_cross_short = hma_bearish and (hma_16[i-1] >= hma_48[i-1])
        
        # === REGIME (Choppiness) ===
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === TREND STRENGTH (ADX) ===
        adx_strong = adx_14[i] > 20.0
        
        # === RSI FILTER (loose thresholds for more trades) ===
        rsi_ok_long = rsi_14[i] > 45.0
        rsi_ok_short = rsi_14[i] < 55.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Long entry: HMA cross + 1d trend + RSI filter
        if hma_cross_long and price_above_hma_1d and rsi_ok_long:
            if adx_strong and is_trending:
                new_signal = POSITION_SIZE_MAX
            elif is_choppy:
                new_signal = POSITION_SIZE_REDUCED
            else:
                new_signal = POSITION_SIZE_BASE
        
        # Short entry: HMA cross + 1d trend + RSI filter
        if hma_cross_short and price_below_hma_1d and rsi_ok_short:
            if adx_strong and is_trending:
                new_signal = -POSITION_SIZE_MAX
            elif is_choppy:
                new_signal = -POSITION_SIZE_REDUCED
            else:
                new_signal = -POSITION_SIZE_BASE
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and no exit signal
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if HMA still bullish
                if hma_bullish:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if HMA still bearish
                if hma_bearish:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
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
        
        # === EXIT ON TREND REVERSAL (1d HMA) ===
        if in_position and position_side > 0:
            if price_below_hma_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_1d:
                new_signal = 0.0
        
        # === EXIT ON HMA REVERSAL ===
        if in_position and position_side > 0 and hma_bearish:
            new_signal = 0.0
        
        if in_position and position_side < 0 and hma_bullish:
            new_signal = 0.0
        
        # === EXIT ON RSI EXTREME (take profit) ===
        if in_position and position_side > 0 and rsi_14[i] > 75.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 25.0:
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
                # Position flip
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