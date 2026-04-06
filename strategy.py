#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray (Bull/Bear Power) with 1-day trend filter and volatility filter.
# Elder Ray measures bull/bear strength relative to EMA. Bull Power = High - EMA, Bear Power = Low - EMA.
# In bull markets, we go long when Bull Power > 0 and rising; in bear markets, short when Bear Power < 0 and falling.
# Daily EMA filter ensures alignment with higher timeframe momentum.
# Volatility filter (ATR-based) avoids choppy markets. Target: 50-150 total trades over 4 years (12-37/year).

name = "elder_ray_6h_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ELDER_RAY_EMA = 13
ATR_PERIOD = 14
ATR_FILTER_MULTIPLIER = 1.0
SIGNAL_SIZE = 0.25

def calculate_ema(close, period):
    """Calculate EMA with Wilder's smoothing"""
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
    if n < 30:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, 50)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Elder Ray components
    ema_13 = calculate_ema(close, ELDER_RAY_EMA)
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # ATR for volatility filter
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(ELDER_RAY_EMA, 50, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if daily EMA not available
        if np.isnan(ema_1d_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely low volatility (chop)
        vol_filter = atr[i] > (np.nanmean(atr[max(0, i-50):i+1]) * ATR_FILTER_MULTIPLIER) if i >= 50 else True
        
        # Trend filter from daily EMA
        uptrend = close[i] > ema_1d_aligned[i]
        downtrend = close[i] < ema_1d_aligned[i]
        
        # Elder Ray signals with trend alignment
        long_signal = bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and uptrend and vol_filter
        short_signal = bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and downtrend and vol_filter
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long when bull power turns negative or trend changes
            if bull_power[i] <= 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short when bear power turns positive or trend changes
            if bear_power[i] >= 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals