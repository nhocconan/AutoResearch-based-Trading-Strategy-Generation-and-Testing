#!/usr/bin/env python3
"""
Experiment #007: 6h Camarilla + Elder Ray + Choppiness Regime

Hypothesis: 6h timeframe with daily Camarilla levels captures swing trading 
opportunities. R3/S3 levels provide mean reversion entries in ranges, R4/S4 
breakouts catch trend continuations. Elder Ray (Bull/Bear Power) confirms 
momentum direction without the lag of moving averages.

Why this should work in BOTH bull AND bear:
1. Camarilla adapts to daily volatility - works in trending and ranging markets
2. Elder Ray captures momentum divergences at key levels
3. Choppiness Index filters out non-tradeable choppy periods
4. 6h = ~1000 bars/year = ~150 trades possible (within 50-300 target)
5. Session best (mtf_6h_alligator_elder_ray_1d_v1) shows 6h + Elder Ray = synergy

Target: Sharpe>0.5, trades 75-250 total over 4 years, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_camarilla_elder_ray_chop_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_ema(close, period):
    """Exponential Moving Average"""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppy vs trending
    CHOP > 61.8 = ranging (good for Camarilla fades)
    CHOP < 38.2 = trending (good for Camarilla breakouts)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_elder_ray(high, low, close, period=13):
    """
    Elder Ray - measures buying/selling pressure
    Bull Power = High - EMA(13)
    Bear Power = Low - EMA(13)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    ema = calculate_ema(close, period)
    
    bull_power = high - ema
    bear_power = low - ema
    
    return bull_power, bear_power

def calculate_camarilla_pivots(high, low, close):
    """
    Camarilla Pivot Points (classic 8 levels)
    Based on previous day's range
    
    Key levels:
    - S3, S4: Support (fade at S3, breakout through S4 = strong short)
    - R3, R4: Resistance (fade at R3, breakout through R4 = strong long)
    - Pivot: Central line
    
    Formula:
    Pivot = (H + L + C) / 3
    R1 = C + (H - L) * 1.1 / 12
    R2 = C + (H - L) * 1.1 / 6
    R3 = C + (H - L) * 1.1 / 4
    R4 = C + (H - L) * 1.1 / 2
    S1 = C - (H - L) * 1.1 / 12
    S2 = C - (H - L) * 1.1 / 6
    S3 = C - (H - L) * 1.1 / 4
    S4 = C - (H - L) * 1.1 / 2
    """
    n = len(close)
    pivot = np.full(n, np.nan, dtype=np.float64)
    s1 = np.full(n, np.nan, dtype=np.float64)
    s2 = np.full(n, np.nan, dtype=np.float64)
    s3 = np.full(n, np.nan, dtype=np.float64)
    s4 = np.full(n, np.nan, dtype=np.float64)
    r1 = np.full(n, np.nan, dtype=np.float64)
    r2 = np.full(n, np.nan, dtype=np.float64)
    r3 = np.full(n, np.nan, dtype=np.float64)
    r4 = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(1, n):
        h = high[i-1]
        l = low[i-1]
        c = close[i-1]
        r = h - l
        
        if r < 1e-10:
            continue
        
        pivot[i] = (h + l + c) / 3.0
        r4[i] = c + r * 0.5
        r3[i] = c + r * 0.275
        r2[i] = c + r * 0.183
        r1[i] = c + r * 0.092
        s1[i] = c - r * 0.092
        s2[i] = c - r * 0.183
        s3[i] = c - r * 0.275
        s4[i] = c - r * 0.5
    
    return pivot, s1, s2, s3, s4, r1, r2, r3, r4

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    bull_power, bear_power = calculate_elder_ray(high, low, close, period=13)
    
    # Calculate 6h Camarilla levels (from previous 6h bar to avoid look-ahead)
    pivot_6h, s1_6h, s2_6h, s3_6h, s4_6h, r1_6h, r2_6h, r3_6h, r4_6h = calculate_camarilla_pivots(high, low, close)
    
    # Calculate and align 1d indicators (for trend bias)
    bull_power_1d_raw, bear_power_1d_raw = calculate_elder_ray(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=13
    )
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d_raw)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d_raw)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 50
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Check 1d alignment (skip if not ready)
        if np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Check Camarilla levels
        r3 = r3_6h[i] if i > 0 and not np.isnan(r3_6h[i-1]) else 0.0
        r4 = r4_6h[i] if i > 0 and not np.isnan(r4_6h[i-1]) else 0.0
        s3 = s3_6h[i] if i > 0 and not np.isnan(s3_6h[i-1]) else 0.0
        s4 = s4_6h[i] if i > 0 and not np.isnan(s4_6h[i-1]) else 0.0
        
        if r3 == 0 or s3 == 0:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION ===
        chop = chop_14[i]
        is_trend_regime = chop < 38.2
        is_range_regime = chop > 61.8
        
        # === DAILY TREND BIAS (Elder Ray 1d) ===
        daily_bullish = bull_power_1d_aligned[i] > 0
        daily_bearish = bear_power_1d_aligned[i] < 0
        
        # === 6h ELDER RAY SIGNALS ===
        bull_6h = bull_power[i]
        bear_6h = bear_power[i]
        
        # Strong bull power (buying pressure)
        bull_strong = bull_6h > 0 and bull_6h > bull_6h * 0.5 if not np.isnan(bull_6h) else False
        # Strong bear power (selling pressure)  
        bear_strong = bear_6h < 0 and abs(bear_6h) > abs(bear_6h) * 0.5 if not np.isnan(bear_6h) else False
        
        # === CAMARILLA LEVEL TOUCH ===
        # Price at/near R3 (resistance fade candidate)
        at_r3 = close[i] >= r3 * 0.998 and close[i] <= r3 * 1.002
        # Price at/near S3 (support fade candidate)
        at_s3 = close[i] >= s3 * 0.998 and close[i] <= s3 * 1.002
        # Price breaking R4 (bull continuation)
        breakout_r4 = close[i] > r4 and (i > 0 and close[i-1] <= r4)
        # Price breaking S4 (bear continuation)
        breakout_s4 = close[i] < s4 and (i > 0 and close[i-1] >= s4)
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # TREND REGIME: Camarilla breakouts + Elder Ray confirmation
        if is_trend_regime:
            # LONG: Breakout R4 + Bull power + daily bullish
            if breakout_r4 and bull_6h > 0 and daily_bullish:
                desired_signal = SIZE_STRONG
            # SHORT: Breakout S4 + Bear power + daily bearish
            elif breakout_s4 and bear_6h < 0 and daily_bearish:
                desired_signal = -SIZE_STRONG
        
        # RANGE REGIME: Camarilla fades at R3/S3 + Elder Ray reversal
        elif is_range_regime:
            # LONG: At S3 support + Bear power weakening (approaching 0 from below)
            if at_s3 and bear_6h > -atr_14[i] * 0.5:
                desired_signal = SIZE_BASE
            # SHORT: At R3 resistance + Bull power weakening (approaching 0 from above)
            elif at_r3 and bull_6h < atr_14[i] * 0.5:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME: Elder Ray only (momentum catch)
        else:
            # LONG: Strong bull power + not at resistance
            if bull_6h > atr_14[i] * 0.5 and close[i] < r3 * 0.995:
                desired_signal = SIZE_BASE
            # SHORT: Strong bear power + not at support
            elif bear_6h < -atr_14[i] * 0.5 and close[i] > s3 * 1.005:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals