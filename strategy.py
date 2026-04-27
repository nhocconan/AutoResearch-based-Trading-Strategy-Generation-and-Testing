#%%
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA(34) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or
            np.isnan(vol_avg_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper + volume spike + daily uptrend
            if (price > highest_20[i] and 
                volume[i] > 1.5 * vol_avg_20[i] and
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower + volume spike + daily downtrend
            elif (price < lowest_20[i] and 
                  volume[i] > 1.5 * vol_avg_20[i] and
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below Donchian lower or trend turns down
            if (price < lowest_20[i] or 
                ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian upper or trend turns up
            if (price > highest_20[i] or 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DonchianBreakout_Volume_DailyTrend_v1"
timeframe = "4h"
leverage = 1.0
#%%