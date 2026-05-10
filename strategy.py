# 1d_LongOnly_TrendFollow_With_WeeklyTrendFilter
# Hypothesis: Long-only trend following on 1d using price above 50-day SMA and weekly trend alignment.
# Weekly trend filter uses 10-week EMA slope to avoid counter-trend trades in bear markets.
# Entry: Price > SMA(50) + weekly uptrend + volume > 1.5x average.
# Exit: Price < SMA(50) or weekly trend turns down.
# Position size: 0.30 when long, 0 when flat.
# Designed for low trade frequency (~10-20 trades/year) to minimize fee drag and survive bear markets.

name = "1d_LongOnly_TrendFollow_With_WeeklyTrendFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 50-day SMA for trend filter
    sma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 10-week EMA for trend filter (weekly)
    ema_10_1w = pd.Series(df_1w['close']).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # Calculate slope of weekly EMA (trend direction)
    # Using 5-period difference to smooth noise
    ema_slope = np.diff(ema_10_1w_aligned, prepend=ema_10_1w_aligned[0]) / 5.0
    ema_slope = np.concatenate([[0], ema_slope[1:]])  # align length
    
    # Volume confirmation (20-day average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    # Warmup: need SMA(50), weekly EMA(10), volume MA(20)
    start_idx = max(50, 10, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(sma_50[i]) or np.isnan(ema_10_1w_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(ema_slope[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Price above 50-day SMA
        price_above_sma = close[i] > sma_50[i]
        
        # Weekly trend up (positive slope)
        weekly_uptrend = ema_slope[i] > 0
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: price above SMA50 + weekly uptrend + volume spike
            if price_above_sma and weekly_uptrend and volume_confirm:
                signals[i] = 0.30
                position = 1
        elif position == 1:
            # Long exit: price crosses below SMA50 or weekly trend turns down
            if close[i] < sma_50[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
    
    return signals