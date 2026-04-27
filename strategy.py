# 12h_TripleConfirmation_Signal
# Hypothesis: Use 1-day EMA(50) for trend, 1-day RSI(14) for momentum, and 1-day volume surge for confirmation.
# Enter long when price > EMA50, RSI > 50, and volume > 1.5x 20-period average.
# Enter short when price < EMA50, RSI < 50, and volume > 1.5x 20-period average.
# Exit when any condition fails.
# Designed for 12h timeframe to capture multi-day trends with volume confirmation.
# Works in both bull and bear markets by following the 1-day trend.
# Target: 15-30 trades/year per symbol (60-120 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend, momentum, and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA(50) for trend
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1-day RSI(14) for momentum
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d.values)
    
    # Calculate 1-day volume MA(20) for volume filter
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need EMA, RSI, and volume MA
    start_idx = max(50, 14, 20)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi_14_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        ema_50 = ema_50_1d_aligned[i]
        rsi = rsi_14_1d_aligned[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        
        # Volume filter: volume > 1.5x 1-day average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions
        if position == 0:
            # Long: price > EMA50, RSI > 50, volume surge
            if close[i] > ema_50 and rsi > 50 and vol_filter:
                signals[i] = size
                position = 1
            # Short: price < EMA50, RSI < 50, volume surge
            elif close[i] < ema_50 and rsi < 50 and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: any condition fails
            if close[i] <= ema_50 or rsi <= 50 or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: any condition fails
            if close[i] >= ema_50 or rsi >= 50 or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_TripleConfirmation_Signal"
timeframe = "12h"
leverage = 1.0