#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour momentum following 4-hour trend with 1-day volume confirmation
# Long when 1h close > 4h EMA(20) and 1d volume > 1.5x 20-day average
# Short when 1h close < 4h EMA(20) and 1d volume > 1.5x 20-day average
# Uses 4h EMA for trend direction, 1d volume for institutional participation confirmation
# 1h timeframe for precise entry timing, targeting 60-150 total trades over 4 years (15-37/year)
# Works in bull markets via trend following, in bear markets via volume-confirmed mean reversion to 4h EMA

name = "1h_EMA20_4hTrend_1dVolumeConfirmation"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data once for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA(20) for trend filter
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Get 1d data once for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume 20-period average for spike detection
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN or outside session
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema20_4h_val = ema20_4h_aligned[i]
        price = close[i]
        vol_ma_1d_val = vol_ma_1d_aligned[i]
        vol_spike = volume[i] > (1.5 * vol_ma_1d_val)
        
        if position == 0:
            # Enter long: price above 4h EMA(20) with volume confirmation
            if price > ema20_4h_val and vol_spike:
                signals[i] = 0.20
                position = 1
            # Enter short: price below 4h EMA(20) with volume confirmation
            elif price < ema20_4h_val and vol_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below 4h EMA(20)
            if price < ema20_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above 4h EMA(20)
            if price > ema20_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals