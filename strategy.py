#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R Extreme with 1d EMA34 trend filter and volume confirmation
# Uses Williams %R(14) to identify oversold/overbought conditions
# Only trade when %R < -80 (oversold) in uptrend or %R > -20 (overbought) in downtrend
# Volume spike (2.0x 20-period average) confirms institutional participation
# 1d EMA34 filter ensures we trade with the daily trend
# Works in bull markets via buying oversold dips in uptrends and bear markets via selling overbought rallies in downtrends
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_WilliamsR_Extreme_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop (MTF Rule #1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA and Williams %R
    
    for i in range(start_idx, n):
        # Williams %R(14) calculation
        if i < 14:
            signals[i] = 0.0
            continue
            
        highest_high = np.max(high[i-13:i+1])
        lowest_low = np.min(low[i-13:i+1])
        curr_close = close[i]
        
        if highest_high == lowest_low:
            williams_r = -50  # avoid division by zero
        else:
            williams_r = (highest_high - curr_close) / (highest_high - lowest_low) * -100
        
        # Volume confirmation: volume > 2.0x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (2.0 * vol_ma_20)
        
        curr_ema_34_1d = ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if volume_spike:
                # Bullish entry: Williams %R < -80 (oversold) AND price above 1d EMA34 (uptrend)
                if williams_r < -80 and curr_close > curr_ema_34_1d:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Williams %R > -20 (overbought) AND price below 1d EMA34 (downtrend)
                elif williams_r > -20 and curr_close < curr_ema_34_1d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when Williams %R rises above -20 (overbought) or price falls below 1d EMA34
            if williams_r > -20 or curr_close < curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Williams %R falls below -80 (oversold) or price rises above 1d EMA34
            if williams_r < -80 or curr_close > curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals