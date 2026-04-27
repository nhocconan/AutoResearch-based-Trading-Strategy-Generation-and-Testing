#!/usr/bin/env python3
"""
1-day Williams %R with 1-week EMA50 trend filter and volume spike filter.
- Long when Williams %R < -80 (oversold) + price > 1w EMA50 (uptrend) + volume spike (>1.5x avg)
- Short when Williams %R > -20 (overbought) + price < 1w EMA50 (downtrend) + volume spike (>1.5x avg)
- Exit when Williams %R crosses back to neutral (-50 for longs, -50 for shorts) or trend reverses
- Williams %R identifies overbought/oversold conditions; EMA50 provides trend filter
- Volume confirmation reduces false signals in low volatility
- Target: 20-40 trades/year (80-160 over 4 years) to minimize fee drag
- Works in bull/bear via trend filter - only trades with higher timeframe trend
"""

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
    
    # Calculate 1d Williams %R(14)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(13, n):
        highest_high[i] = np.max(high[i-13:i+1])
        lowest_low[i] = np.min(low[i-13:i+1])
    
    williams_r = np.full(n, np.nan)
    for i in range(13, n):
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            williams_r[i] = -50  # neutral when no range
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for Williams %R and EMA
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: Williams %R oversold + above 1w EMA50 + volume spike
            if (williams_r[i] < -80 and close[i] > ema50_1w_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought + below 1w EMA50 + volume spike
            elif (williams_r[i] > -20 and close[i] < ema50_1w_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Williams %R crosses above -50 (mean reversion complete) OR below 1w EMA50 (trend change)
            if (williams_r[i] > -50 or close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -50 (mean reversion complete) OR above 1w EMA50 (trend change)
            if (williams_r[i] < -50 or close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsR14_1wEMA50_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0