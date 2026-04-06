#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Alligator + Elder Ray power with 1-day trend filter.
# The Alligator (Jaw/Teeth/Lips) identifies trend absence/presence via SMAs.
# Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures trend strength.
# Combined with 1-day EMA50 trend filter to avoid counter-trend trades.
# Works in bull/bear by only taking trades when Alligator is awake (trending) and Elder Ray confirms direction.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "exp_13251_6w_alligator_elder_ray_1d_ema_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_JAW = 13    # Smoothed SMA (13, 8)
ALLIGATOR_TEETH = 8   # Smoothed SMA (8, 5)
ALLIGATOR_LIPS = 5    # Smoothed SMA (5, 3)
ELDER_RAY_EMA = 13
TREND_EMA = 50        # 1-day EMA for trend filter
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def smooth_sma(series, period):
    """Smoothed SMA: SMA of SMA"""
    sma1 = pd.Series(series).rolling(window=period, min_periods=period).mean()
    sma2 = sma1.rolling(window=period, min_periods=period).mean()
    return sma2.values

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, TREND_EMA)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Alligator components (smoothed SMAs)
    jaw = smooth_sma(medium, ALLIGATOR_JAW) if 'medium' in locals() else smooth_sma(close, ALLIGATOR_JAW)
    teeth = smooth_sma(close, ALLIGATOR_TEETH)
    lips = smooth_sma(close, ALLIGATOR_LIPS)
    
    # Elder Ray components
    ema13 = calculate_ema(close, ELDER_RAY_EMA)
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # ATR for stoploss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Warmup: need enough data for Alligator (slowest is Jaw: 13*2=26 bars for smoothing)
    start = max(ALLIGATOR_JAW*2, ALLIGATOR_TEETH*2, ALLIGATOR_LIPS*2, ELDER_RAY_EMA, TREND_EMA, ATR_PERIOD) + 5
    
    for i in range(start, n):
        # Skip if daily EMA not available
        if np.isnan(ema_1d_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Alligator: check if awake (jaws, teeth, lips not intertwined)
        # Alligator is asleep when jaws ~ teeth ~ lips (all close together)
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        # Calculate spread between extreme values
        max_val = max(jaw_val, teeth_val, lips_val)
        min_val = min(jaw_val, teeth_val, lips_val)
        spread = max_val - min_val
        avg_price = (jaw_val + teeth_val + lips_val) / 3
        
        # Normalized spread as percentage of price
        if avg_price != 0:
            alligator_awake = (spread / avg_price) > 0.005  # 0.5% threshold
        else:
            alligator_awake = False
        
        # Elder Ray: bull/bear power strength
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        
        # Trend filter from daily EMA
        uptrend_filter = close[i] > ema_1d_aligned[i]
        downtrend_filter = close[i] < ema_1d_aligned[i]
        
        # Entry conditions
        if position == 0:
            # Long: Alligator awake + bull power positive + uptrend filter
            if alligator_awake and bull_val > 0 and uptrend_filter:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Short: Alligator awake + bear power positive + downtrend filter
            elif alligator_awake and bear_val > 0 and downtrend_filter:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals