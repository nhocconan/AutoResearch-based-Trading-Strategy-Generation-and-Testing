#!/usr/bin/env python3
"""
Experiment #104: 4h Primary + 12h/1d HTF — KAMA Fisher Regime Strategy

Hypothesis: Previous 4h strategies failed because they used static indicators that
don't adapt to changing volatility regimes. This strategy combines:

1) KAMA (Kaufman Adaptive MA) - adapts smoothing based on market efficiency
   - Fast in trends, slow in chop (unlike fixed EMA/HMA)
   - Proven in #079 with ETH Sharpe +0.755

2) Ehlers Fisher Transform - normalized oscillator (-1 to +1)
   - Better reversal detection than RSI in bear markets
   - Entry when Fisher crosses -0.8 (long) or +0.8 (short)

3) Choppiness Index regime switch
   - CHOP > 55 = range mode (mean revert at KAMA extremes)
   - CHOP < 45 = trend mode (follow 12h KAMA direction)

4) Donchian Channel breakout confirmation
   - Long: price > Donchian(20) high + 12h KAMA bullish
   - Short: price < Donchian(20) low + 12h KAMA bearish

5) 12h HTF for macro bias (not 1w - too laggy for 4h)
   - 12h KAMA slope determines allowed direction
   - Prevents counter-trend trades in bear markets

6) Conservative sizing with ATR trailing stop
   - Base: 0.25, Max: 0.30 with confluence
   - Stop: 2.5*ATR trailing from entry

Why this should beat Sharpe=0.486 baseline:
- KAMA adapts to 2022 crash volatility better than HMA
- Fisher Transform catches bear market reversals (RSI fails here)
- 12h HTF less laggy than 1w for 4h entries
- Donchian breakout = fewer false signals, higher win rate
- Target: 25-45 trades/year, Sharpe > 0.6 on ALL symbols

Position size: 0.25 base, 0.30 max
Stoploss: 2.5*ATR trailing
Timeframe: 4h (proven best for crypto futures)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_fisher_chop_donchian_12h1d_v1"
timeframe = "4h"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts smoothing based on market efficiency ratio.
    Fast in trends, slow in chop.
    """
    close_s = pd.Series(close)
    
    # Efficiency Ratio (ER)
    change = close_s.diff(er_period).abs()
    volatility = close_s.diff().abs().rolling(window=er_period, min_periods=er_period).sum()
    er = change / (volatility + 1e-10)
    er = er.fillna(0.0)
    
    # Smoothing constant
    sc = (er * (2.0/(fast_period+1) - 2.0/(slow_period+1)) + 2.0/(slow_period+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price to -1 to +1 range for better reversal detection.
    """
    # Calculate typical price
    typical = (high + low + close) / 3.0
    
    # Normalize to -1 to +1
    highest = pd.Series(typical).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(typical).rolling(window=period, min_periods=period).min().values
    
    normalized = np.zeros(len(close))
    for i in range(period, len(close)):
        range_val = highest[i] - lowest[i]
        if range_val > 1e-10:
            normalized[i] = 2.0 * (typical[i] - lowest[i]) / range_val - 1.0
        else:
            normalized[i] = 0.0
    
    # Clip to avoid log errors
    normalized = np.clip(normalized, -0.999, 0.999)
    
    # Fisher transform
    fisher = np.zeros(len(close))
    for i in range(period, len(close)):
        fisher[i] = 0.5 * np.log((1.0 + normalized[i]) / (1.0 - normalized[i]) + 1e-10)
        if i > period:
            fisher[i] = 0.67 * fisher[i] + 0.33 * fisher[i-1]  # smoothing
    
    return fisher

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
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
    chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    chop = np.nan_to_num(chop, nan=50.0)
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h KAMA for trend bias
    kama_12h = calculate_kama(df_12h['close'].values, er_period=10)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # Calculate 12h KAMA slope
    kama_12h_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(kama_12h_aligned[i]) and not np.isnan(kama_12h_aligned[i-1]) and kama_12h_aligned[i-1] != 0:
            kama_12h_slope[i] = (kama_12h_aligned[i] - kama_12h_aligned[i-1]) / kama_12h_aligned[i-1] * 100
        else:
            kama_12h_slope[i] = 0.0
    
    # Calculate 1d KAMA for macro filter
    kama_1d = calculate_kama(df_1d['close'].values, er_period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    fisher = calculate_fisher_transform(high, low, close, period=9)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    kama_4h = calculate_kama(close, er_period=10)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(kama_12h_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(fisher[i]) or np.isnan(chop_14[i]) or np.isnan(kama_4h[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(kama_1d_aligned[i]):
            continue
        
        # === HTF TREND BIAS (12h KAMA) ===
        price_above_kama_12h = close[i] > kama_12h_aligned[i]
        price_below_kama_12h = close[i] < kama_12h_aligned[i]
        kama_12h_slope_positive = kama_12h_slope[i] > 0.1
        kama_12h_slope_negative = kama_12h_slope[i] < -0.1
        
        # 1d KAMA macro filter
        price_above_kama_1d = close[i] > kama_1d_aligned[i] if not np.isnan(kama_1d_aligned[i]) else True
        price_below_kama_1d = close[i] < kama_1d_aligned[i] if not np.isnan(kama_1d_aligned[i]) else True
        
        # === CHOPPINESS REGIME ===
        chop_trending = chop_14[i] < 45.0  # trending market
        chop_ranging = chop_14[i] > 55.0  # ranging market
        chop_neutral = not chop_trending and not chop_ranging
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1]  # break above previous high
        donchian_breakout_short = close[i] < donchian_lower[i-1]  # break below previous low
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -0.8  # extreme oversold
        fisher_overbought = fisher[i] > 0.8  # extreme overbought
        fisher_cross_up = fisher[i] > -0.5 and fisher[i-1] <= -0.5 if i > 0 else False
        fisher_cross_down = fisher[i] < 0.5 and fisher[i-1] >= 0.5 if i > 0 else False
        
        # === KAMA POSITION ===
        price_above_kama_4h = close[i] > kama_4h[i]
        price_below_kama_4h = close[i] < kama_4h[i]
        kama_distance = (close[i] - kama_4h[i]) / kama_4h[i] * 100 if kama_4h[i] != 0 else 0
        kama_far_below = kama_distance < -2.0  # price far below KAMA (mean revert long)
        kama_far_above = kama_distance > 2.0  # price far above KAMA (mean revert short)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        long_bias = price_above_kama_12h and kama_12h_slope_positive and price_above_kama_1d
        
        if chop_trending:
            # TREND MODE: Follow HTF trend with breakout confirmation
            if long_bias:
                if donchian_breakout_long or fisher_cross_up:
                    new_signal = POSITION_SIZE_BASE
                    if donchian_breakout_long and fisher_cross_up:
                        new_signal = POSITION_SIZE_MAX
        elif chop_ranging:
            # RANGE MODE: Mean revert at KAMA extremes
            if kama_far_below and (fisher_oversold or fisher_cross_up):
                new_signal = POSITION_SIZE_BASE
                if fisher_oversold:
                    new_signal = POSITION_SIZE_MAX
        else:
            # NEUTRAL: Require strong confluence
            if long_bias and fisher_cross_up and price_above_kama_4h:
                new_signal = POSITION_SIZE_BASE
        
        # --- SHORT ENTRY ---
        short_bias = price_below_kama_12h and kama_12h_slope_negative and price_below_kama_1d
        
        if chop_trending:
            # TREND MODE: Follow HTF trend with breakout confirmation
            if short_bias:
                if donchian_breakout_short or fisher_cross_down:
                    new_signal = -POSITION_SIZE_BASE
                    if donchian_breakout_short and fisher_cross_down:
                        new_signal = -POSITION_SIZE_MAX
        elif chop_ranging:
            # RANGE MODE: Mean revert at KAMA extremes
            if kama_far_above and (fisher_overbought or fisher_cross_down):
                new_signal = -POSITION_SIZE_BASE
                if fisher_overbought:
                    new_signal = -POSITION_SIZE_MAX
        else:
            # NEUTRAL: Require strong confluence
            if short_bias and fisher_cross_down and price_below_kama_4h:
                new_signal = -POSITION_SIZE_BASE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            # Hold long if Fisher not overbought
            if position_side > 0 and fisher[i] < 0.7:
                new_signal = signals[i-1] if i > 0 and signals[i-1] != 0 else POSITION_SIZE_BASE
            # Hold short if Fisher not oversold
            elif position_side < 0 and fisher[i] > -0.7:
                new_signal = signals[i-1] if i > 0 and signals[i-1] != 0 else -POSITION_SIZE_BASE
        
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
        
        # === EXIT ON TREND CHANGE ===
        if in_position and position_side > 0:
            if price_below_kama_12h and kama_12h_slope_negative:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_kama_12h and kama_12h_slope_positive:
                new_signal = 0.0
        
        # === EXIT ON FISHER EXTREME (take profit) ===
        if in_position and position_side > 0 and fisher[i] > 1.2:
            new_signal = 0.0
        
        if in_position and position_side < 0 and fisher[i] < -1.2:
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