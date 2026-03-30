#!/usr/bin/env python3
"""
Experiment #028: 6h Elder Ray + Volume + 1d EMA Trend

HYPOTHESIS: Elder Ray (Bull/Bear Power) measures institutional buying/selling pressure
relative to a smoothed average. Unlike RSI or Williams %R which are bounded oscillators,
Elder Ray captures the MAGNITUDE of force behind moves. Combined with 1d EMA for trend
and volume for confirmation, this catches genuine breakouts while avoiding whipsaws.

WHY IT WORKS IN BULL AND BEAR:
- Bull Power > 0 in uptrends = institutional buying confirms rallies
- Bear Power < 0 in downtrends = institutional selling confirms selloffs  
- Divergence signals exhaustion BEFORE reversals
- Symmetric: works on both long and short sides

WHY 6h: Slower than 4h reduces fee drag, faster than 12h catches more setups.
6h bars capture institutional sessions without noise.

WHY NOVEL: Elder Ray is DIFFERENT from RSI/WR/TRIX - it measures absolute power
above/below EMA, not relative position. Not tried in previous experiments.

TARGET: 75-150 total trades over 4 years (19-37/year). HARD MAX: 200.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_elder_ray_vol_1d_ema_v1"
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

def calculate_ema(values, period, min_periods=None):
    """Exponential Moving Average"""
    if min_periods is None:
        min_periods = period
    return pd.Series(values).ewm(span=period, min_periods=min_periods, adjust=False).mean().values

def calculate_elder_ray(high, low, close, ema_period=13):
    """
    Elder Ray: Measures buying/selling pressure relative to EMA
    Bull Power = High - EMA (positive = buying pressure)
    Bear Power = Low - EMA (negative = selling pressure)
    """
    n = len(close)
    ema = calculate_ema(close, ema_period, min_periods=ema_period)
    
    bull_power = np.zeros(n, dtype=np.float64)
    bear_power = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        bull_power[i] = high[i] - ema[i]
        bear_power[i] = low[i] - ema[i]
    
    return bull_power, bear_power, ema

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA for trend direction (26-period = one month)
    ema_1d = calculate_ema(df_1d['close'].values, 26, min_periods=26)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Local 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    bull_power, bear_power, ema_local = calculate_elder_ray(high, low, close, ema_period=13)
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # EMA of Elder Ray for smoothing (avoid noise)
    bull_power_smooth = pd.Series(bull_power).ewm(span=3, min_periods=3, adjust=False).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=3, min_periods=3, adjust=False).mean().values
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 100  # Need enough for all indicators
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA26) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_1d_aligned[i]
        
        # === ELDER RAY SIGNALS ===
        bull = bull_power_smooth[i]
        bear = bear_power_smooth[i]
        
        # Strong institutional buying: Bull Power breaks above threshold
        strong_bull = bull > atr_14[i] * 0.3
        
        # Strong institutional selling: Bear Power breaks below threshold
        strong_bear = bear < -atr_14[i] * 0.3
        
        # Volume confirmation
        vol_confirm = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Strong bull power + uptrend + volume ===
            # Institutional buying pressure pushing price above EMA
            if strong_bull and price_above_1d_ema:
                if vol_confirm:
                    desired_signal = SIZE
            
            # === SHORT: Strong bear power + downtrend + volume ===
            # Institutional selling pressure pushing price below EMA
            if strong_bear and price_below_1d_ema:
                if vol_confirm:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
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
        
        # === TIME-BASED EXIT (hold at least 8 bars = 2 days) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 8:
            # Exit if Elder Ray reverses
            if position_side > 0 and bear < -atr_14[i] * 0.1:
                desired_signal = 0.0
            if position_side < 0 and bull > atr_14[i] * 0.1:
                desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
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
        
        signals[i] = desired_signal
    
    return signals