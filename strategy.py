# 6h_Breakout_1wEMA100_Trend_VolumeSurge
# Strategy: 6h breakout above 1w EMA100 with volume surge and 1d trend filter.
# Works in bull markets (breakouts above rising EMA100) and bear markets (breakdowns below falling EMA100).
# Volume surge filters false breakouts; 1d trend ensures alignment with higher timeframe momentum.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Breakout_1wEMA100_Trend_VolumeSurge"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly EMA100 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    weekly_close = df_1w['close'].values
    ema_100_1w = pd.Series(weekly_close).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_100_1w)
    
    # Daily trend: price above/below EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    daily_close = df_1d['close'].values
    ema_50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume surge: current volume > 3.0x 24-period average
    vol_ma_24 = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma_24[i] = np.mean(volume[i-24:i])
    volume_surge = volume > (3.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 12  # ~3 days to prevent overtrading
    
    start_idx = 100  # EMA100 needs 100 bars
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_100_1w_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine weekly trend direction (slope of EMA100)
        if i >= 101:
            weekly_trend_up = ema_100_1w_aligned[i] > ema_100_1w_aligned[i-1]
            weekly_trend_down = ema_100_1w_aligned[i] < ema_100_1w_aligned[i-1]
        else:
            weekly_trend_up = False
            weekly_trend_down = False
        
        # Determine daily trend direction
        daily_trend_up = close[i] > ema_50_1d_aligned[i]
        daily_trend_down = close[i] < ema_50_1d_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price breaks above weekly EMA100 with volume surge in weekly uptrend
            if (close[i] > ema_100_1w_aligned[i] and 
                weekly_trend_up and 
                volume_surge[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below weekly EMA100 with volume surge in weekly downtrend
            elif (close[i] < ema_100_1w_aligned[i] and 
                  weekly_trend_down and 
                  volume_surge[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls back below weekly EMA100 or weekly trend changes to down
            if close[i] < ema_100_1w_aligned[i] or not weekly_trend_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises back above weekly EMA100 or weekly trend changes to up
            if close[i] > ema_100_1w_aligned[i] or not weekly_trend_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
# 6h_Breakout_1wEMA100_Trend_VolumeSurge
# Strategy: 6h breakout above 1w EMA100 with volume surge and 1d trend filter.
# Works in bull markets (breakouts above rising EMA100) and bear markets (breakdowns below falling EMA100).
# Volume surge filters false breakouts; 1d trend ensures alignment with higher timeframe momentum.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Breakout_1wEMA100_Trend_VolumeSurge"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly EMA100 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    weekly_close = df_1w['close'].values
    ema_100_1w = pd.Series(weekly_close).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_100_1w)
    
    # Daily trend: price above/below EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    daily_close = df_1d['close'].values
    ema_50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume surge: current volume > 3.0x 24-period average
    vol_ma_24 = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma_24[i] = np.mean(volume[i-24:i])
    volume_surge = volume > (3.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 12  # ~3 days to prevent overtrading
    
    start_idx = 100  # EMA100 needs 100 bars
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_100_1w_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine weekly trend direction (slope of EMA100)
        if i >= 101:
            weekly_trend_up = ema_100_1w_aligned[i] > ema_100_1w_aligned[i-1]
            weekly_trend_down = ema_100_1w_aligned[i] < ema_100_1w_aligned[i-1]
        else:
            weekly_trend_up = False
            weekly_trend_down = False
        
        # Determine daily trend direction
        daily_trend_up = close[i] > ema_50_1d_aligned[i]
        daily_trend_down = close[i] < ema_50_1d_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price breaks above weekly EMA100 with volume surge in weekly uptrend
            if (close[i] > ema_100_1w_aligned[i] and 
                weekly_trend_up and 
                volume_surge[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below weekly EMA100 with volume surge in weekly downtrend
            elif (close[i] < ema_100_1w_aligned[i] and 
                  weekly_trend_down and 
                  volume_surge[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls back below weekly EMA100 or weekly trend changes to down
            if close[i] < ema_100_1w_aligned[i] or not weekly_trend_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises back above weekly EMA100 or weekly trend changes to up
            if close[i] > ema_100_1w_aligned[i] or not weekly_trend_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals