#!/usr/bin/env python3
name = "6h_ChaikinMoneyFlow_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Money Flow Multiplier and Volume for Chaikin Money Flow (CMF)
    mfm = np.where((high - low) != 0, ((close - low) - (high - close)) / (high - low), 0)
    mfv = mfm * volume
    
    # CMF(20) - sum of MFV over 20 periods / sum of volume over 20 periods
    mfv_sum = np.full(n, np.nan)
    vol_sum = np.full(n, np.nan)
    for i in range(20, n):
        mfv_sum[i] = np.sum(mfv[i-20:i])
        vol_sum[i] = np.sum(volume[i-20:i])
    cmf = np.divide(mfv_sum, vol_sum, out=np.full(n, np.nan), where=vol_sum!=0)
    
    # Daily trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Daily EMA50 trend
    ema_50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    trend_up = close > ema_50_1d_aligned
    trend_down = close < ema_50_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 3  # ~18 hours (3*6h) to prevent overtrading
    
    start_idx = max(20, 50)  # Ensure enough data for CMF and EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(cmf[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine trend direction
        trending_up = trend_up[i]
        trending_down = trend_down[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: CMF > 0.15 with volume accumulation in daily uptrend
            if (cmf[i] > 0.15 and 
                trending_up):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: CMF < -0.15 with volume distribution in daily downtrend
            elif (cmf[i] < -0.15 and 
                  trending_down):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: CMF falls below 0 or daily trend changes to down
            if cmf[i] < 0 or not trending_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: CMF rises above 0 or daily trend changes to up
            if cmf[i] > 0 or not trending_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Chaikin Money Flow (CMF) measures institutional buying/selling pressure through volume-weighted accumulation/distribution. 
# On 6h timeframe, CMF > 0.15 indicates strong buying pressure, CMF < -0.15 indicates strong selling pressure. 
# Daily EMA50 filter ensures alignment with higher timeframe trend to avoid counter-trend trades. 
# The strategy works in bull markets (long when CMF>0.15 in daily uptrend) and bear markets (short when CMF<-0.15 in daily downtrend). 
# Cooldown period (3 bars = 18h) prevents overtrading, targeting 50-150 total trades over 4 years (12-37/year) to minimize fee drag. 
# Discrete position sizing (0.25) balances risk and return while reducing fee churn from frequent position changes.