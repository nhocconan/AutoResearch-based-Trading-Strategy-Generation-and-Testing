#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d Trend Filter and Volume Spike
# - Uses Williams %R from daily timeframe to identify overbought/oversold conditions
# - Long when %R crosses above -80 (oversold) with 1d uptrend and volume spike
# - Short when %R crosses below -20 (overbought) with 1d downtrend and volume spike
# - Works in bull/bear by using 1d trend filter to align with higher timeframe momentum
# - Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag on 12h timeframe

name = "12h_WilliamsR_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Williams %R and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    lookback = 14
    williams_r = np.full(len(close_1d), np.nan)
    
    for i in range(lookback, len(close_1d)):
        highest_high = np.max(high_1d[i-lookback:i])
        lowest_low = np.min(low_1d[i-lookback:i])
        close_val = close_1d[i]
        if highest_high != lowest_low:
            williams_r[i] = ((highest_high - close_val) / (highest_high - lowest_low)) * -100
        else:
            williams_r[i] = -50  # neutral when no range
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align Williams %R and EMA to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below (exiting oversold) with 1d uptrend + volume spike
            wr_cross_up = (williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80)
            long_cond = wr_cross_up and (ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]) and volume_spike[i]
            
            # Short: Williams %R crosses below -20 from above (exiting overbought) with 1d downtrend + volume spike
            wr_cross_down = (williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20)
            short_cond = wr_cross_down and (ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]) and volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -20 (entering overbought)
            if williams_r_aligned[i] > -20 and williams_r_aligned[i-1] <= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -80 (entering oversold)
            if williams_r_aligned[i] < -80 and williams_r_aligned[i-1] >= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals