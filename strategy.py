#!/usr/bin/env python3
"""
Experiment #007: 6h Elder Ray + Choppiness Index Regime

HYPOTHESIS: Elder Ray (Bull/Bear Power) measures institutional momentum.
- Bull Power = High - EMA(13) — positive = buying pressure
- Bear Power = Low - EMA(13) — negative = selling pressure

WHY BOTH BULL AND BEAR: Symmetric momentum signals.
- In UPTREND: buy when Bear Power dips below zero (pullback exhaustion)
- In DOWNTREND: short when Bull Power rises above zero (rally exhaustion)
- Choppiness Index < 38.2 confirms trending conditions (avoid range markets)

WHY 6h: Balance between 4h (too many trades) and 12h (too few).
Slower than 4h = fewer trades = less fee drag, but faster than 12h = more opportunities.

TARGET: 75-150 total trades over 4 years = 19-37/year. HARD MAX: 200.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_elder_ray_chop_regime_v1"
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

def calculate_choppiness_index(high, low, close, period=14):
    """Choppiness Index — lower = trending, higher = ranging"""
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest > lowest:
            sum_tr = 0.0
            for j in range(i - period + 1, i + 1):
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
                sum_tr += tr
            
            chop[i] = 100 * (np.log(sum_tr) / np.log(highest - lowest)) / np.log(period)
    
    return chop

def calculate_ema(values, period, min_periods=None):
    """Exponential Moving Average"""
    if min_periods is None:
        min_periods = period
    return pd.Series(values).ewm(span=period, min_periods=min_periods, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA200 for trend direction (slower EMA = longer-term trend)
    ema_200_1d = calculate_ema(df_1d['close'].values, 200, min_periods=200)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === Local 6h Elder Ray (classic period=13) ===
    ema_13 = calculate_ema(close, 13, min_periods=13)
    bull_power = high - ema_13  # Buying pressure
    bear_power = low - ema_13   # Selling pressure
    
    # Smooth power values with EMA of power
    bull_power_smooth = calculate_ema(bull_power, 5, min_periods=5)
    bear_power_smooth = calculate_ema(bear_power, 5, min_periods=5)
    
    # Choppiness Index for regime detection
    chop = calculate_choppiness_index(high, low, close, period=14)
    
    # ATR for stoploss
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume ratio (20-bar MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === ELDER RAY Z-SCORE (normalized for better entry) ===
    # Calculate z-score of power values over lookback
    lookback = 50
    
    bull_mean = pd.Series(bull_power).rolling(window=lookback, min_periods=lookback).mean().values
    bull_std = pd.Series(bull_power).rolling(window=lookback, min_periods=lookback).std().values
    bull_z = (bull_power - bull_mean) / np.where(bull_std > 1e-10, bull_std, 1e-10)
    
    bear_mean = pd.Series(bear_power).rolling(window=lookback, min_periods=lookback).mean().values
    bear_std = pd.Series(bear_power).rolling(window=lookback, min_periods=lookback).std().values
    bear_z = (bear_power - bear_mean) / np.where(bear_std > 1e-10, bear_std, 1e-10)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    chop_exit_bars = 0  # Counter for chop confirmation before exit
    
    warmup = 250  # Need enough for EMA200 alignment buffer
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if 1d EMA not aligned
        if np.isnan(ema_200_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if chop not ready
        if np.isnan(chop[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA200) ===
        in_uptrend = close[i] > ema_200_1d_aligned[i]
        in_downtrend = close[i] < ema_200_1d_aligned[i]
        
        # === REGIME: Choppiness < 38.2 = trending ===
        is_trending = chop[i] < 38.2
        
        # Volume confirmation
        vol_confirm = vol_ratio[i] > 1.2
        
        # === ELDER RAY SIGNALS ===
        # Bull Power crossed above zero (buying pressure emerging)
        bull_cross_up = bull_power[i] > 0 and (i > 0 and bull_power[i-1] <= 0)
        # Bull Power positive and rising
        bull_momentum = bull_power[i] > 0 and bull_power[i] > bull_power[i-1]
        
        # Bear Power crossed below zero (selling pressure exhausting)
        bear_cross_down = bear_power[i] < 0 and (i > 0 and bear_power[i-1] >= 0)
        # Bear Power negative and falling
        bear_momentum = bear_power[i] < 0 and bear_power[i] < bear_power[i-1]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY: In uptrend, Bear Power shows exhaustion ===
            # Bear Power crossed below zero OR very negative + bounce starting
            if in_uptrend and is_trending and vol_confirm:
                # Bear Power crossed below zero (selling exhausted)
                if bear_cross_down:
                    desired_signal = SIZE
                # Or Bear Power very negative (z-score < -1.5) with Bull Power turning up
                elif bear_z[i] < -1.5 and bull_cross_up:
                    desired_signal = SIZE
            
            # === SHORT ENTRY: In downtrend, Bull Power shows exhaustion ===
            if in_downtrend and is_trending and vol_confirm:
                # Bull Power crossed above zero (buying exhausted)
                if bull_cross_up:
                    desired_signal = -SIZE
                # Or Bull Power very positive (z-score > 1.5) with Bear Power turning down
                elif bull_z[i] > 1.5 and bear_cross_down:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR — slightly wider for 6h) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            if low[i] < trailing_stop:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            if high[i] > trailing_stop:
                desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 bars (18h) to avoid fee churn ===
        bars_held = i - entry_bar
        if in_position and bars_held < 3:
            # Keep position, don't exit early
            desired_signal = position_side * SIZE
        
        # === TAKE PROFIT: Exit when power reverts to mean ===
        if in_position and bars_held >= 3:
            if position_side > 0:
                # Exit if Bull Power turns negative (momentum fading)
                if bull_power[i] < 0:
                    desired_signal = 0.0
            if position_side < 0:
                # Exit if Bear Power turns positive (momentum fading)
                if bear_power[i] > 0:
                    desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals