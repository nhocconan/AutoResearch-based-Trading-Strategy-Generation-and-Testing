# [75982] 12h Donchian(20) breakout + 1d EMA34 trend + volume confirmation  
# Hypothesis: Breakouts from daily price extremes (Donchian 20) on 12h,  
# filtered by daily EMA34 trend (trend filter) and volume spike.  
# Works in bull/bear: Trend filter avoids counter-trend breakouts;  
# volume ensures breakout conviction.  
# Target: 15-30 trades/year (60-120 total) on 12h.  

#!/usr/bin/env python3
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
    
    # Load daily data for Donchian(20) and EMA34 - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    
    # Calculate Donchian(20) channels from daily data
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    upper_20 = pd.Series(high_daily).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_daily).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily EMA34 trend
    close_daily = df_daily['close'].values
    ema_34 = pd.Series(close_daily).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 12h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_daily, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_daily, lower_20)
    ema_34_aligned = align_htf_to_ltf(prices, df_daily, ema_34)
    
    # Pre-calculate volume average (20-period) on 12h
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC) - optional filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC (reduce noise outside active hours)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper Donchian(20) + daily uptrend + volume
            if (close[i] > upper_20_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian(20) + daily downtrend + volume
            elif (close[i] < lower_20_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to the opposite Donchian channel
            if position == 1:
                if close[i] < lower_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > upper_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12H_Donchian20_DailyEMA34_Volume"
timeframe = "12h"
leverage = 1.0