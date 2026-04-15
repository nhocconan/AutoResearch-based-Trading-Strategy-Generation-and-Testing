#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h EMA50 + Volume Confirmation
# Uses bull power (EMA13 - Low) and bear power (High - EMA13) to measure buying/selling pressure.
# Trades in direction of 12h EMA50 trend when power confirms and volume > 1.5x 20-bar median.
# Works in bull markets (buy on bull power) and bear markets (sell on bear power).
# Target: 50-150 total trades over 4 years = 12-37/year.
# Timeframe: 6h, HTF: 12h

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA13 for Elder Ray calculation
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = Close - EMA13 (buying pressure)
    # Bear Power = EMA13 - Close (selling pressure) - but we'll compute directly as High/EMA13 and Low/EMA13
    bull_power = close - ema13  # Close - EMA13
    bear_power = ema13 - close  # EMA13 - Close
    
    # Load 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(13, n):  # Start after EMA13 warmup
        # Skip if EMA50 not available
        if np.isnan(ema50_12h_aligned[i]):
            continue
        
        # Determine trend direction from 12h EMA50
        # Uptrend if current close > EMA50, downtrend if current close < EMA50
        is_uptrend = close[i] > ema50_12h_aligned[i]
        is_downtrend = close[i] < ema50_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.5x median of last 20 bars
        vol_median = np.median(volume[max(0, i-20):i+1]) if i >= 20 else np.median(volume[:i+1])
        volume_confirm = volume[i] > 1.5 * vol_median
        
        # Long entry: uptrend + bull power positive + volume confirmation
        if is_uptrend and bull_power[i] > 0 and volume_confirm and position <= 0:
            position = 1
            signals[i] = base_size
        
        # Short entry: downtrend + bear power positive + volume confirmation
        elif is_downtrend and bear_power[i] > 0 and volume_confirm and position >= 0:
            position = -1
            signals[i] = -base_size
        
        # Exit: trend change or power divergence
        elif position == 1 and (not is_uptrend or bull_power[i] <= 0):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not is_downtrend or bear_power[i] <= 0):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0