#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA trend + volume confirmation
# In trending markets (1d EMA50 slope > 0), buy breakouts above 20-period high
# In ranging markets (1d EMA50 slope <= 0), fade at 20-period low with volume
# Uses volume > 1.5x average for confirmation
# Targets 75-200 total trades over 4 years (19-50/year) with strict breakout conditions

name = "4h_donchian_1d_ema_vol_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d EMA50 for trend direction
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50
    ema50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate EMA50 slope (trend strength)
    ema50_slope = np.zeros_like(ema50)
    ema50_slope[1:] = ema50[1:] - ema50[:-1]
    
    # Align EMA50 and slope to 4h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    ema50_slope_aligned = align_htf_to_ltf(prices, df_1d, ema50_slope)
    
    # 4h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema50_aligned[i]) or np.isnan(ema50_slope_aligned[i]) or 
            np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend: up (slope > 0), down (slope < 0), or flat
        trend_up = ema50_slope_aligned[i] > 0
        trend_down = ema50_slope_aligned[i] < 0
        
        if position == 1:  # long position
            # Exit: price touches or crosses below 20-period low
            if close[i] <= low_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price touches or crosses above 20-period high
            if close[i] >= high_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries based on trend
            if trend_up:
                # Uptrend: buy breakout above 20-period high with volume
                if close[i] > high_roll[i] and volume[i] > volume_threshold[i]:
                    signals[i] = 0.25
                    position = 1
            elif trend_down:
                # Downtrend: sell breakdown below 20-period low with volume
                if close[i] < low_roll[i] and volume[i] > volume_threshold[i]:
                    signals[i] = -0.25
                    position = -1
            # In flat markets (no trend), wait for clearer signal
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA trend + volume confirmation
# In trending markets (1d EMA50 slope > 0), buy breakouts above 20-period high
# In ranging markets (1d EMA50 slope <= 0), fade at 20-period low with volume
# Uses volume > 1.5x average for confirmation
# Targets 75-200 total trades over 4 years (19-50/year) with strict breakout conditions

name = "4h_donchian_1d_ema_vol_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d EMA50 for trend direction
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50
    ema50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate EMA50 slope (trend strength)
    ema50_slope = np.zeros_like(ema50)
    ema50_slope[1:] = ema50[1:] - ema50[:-1]
    
    # Align EMA50 and slope to 4h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    ema50_slope_aligned = align_htf_to_ltf(prices, df_1d, ema50_slope)
    
    # 4h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema50_aligned[i]) or np.isnan(ema50_slope_aligned[i]) or 
            np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend: up (slope > 0), down (slope < 0), or flat
        trend_up = ema50_slope_aligned[i] > 0
        trend_down = ema50_slope_aligned[i] < 0
        
        if position == 1:  # long position
            # Exit: price touches or crosses below 20-period low
            if close[i] <= low_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price touches or crosses above 20-period high
            if close[i] >= high_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries based on trend
            if trend_up:
                # Uptrend: buy breakout above 20-period high with volume
                if close[i] > high_roll[i] and volume[i] > volume_threshold[i]:
                    signals[i] = 0.25
                    position = 1
            elif trend_down:
                # Downtrend: sell breakdown below 20-period low with volume
                if close[i] < low_roll[i] and volume[i] > volume_threshold[i]:
                    signals[i] = -0.25
                    position = -1
            # In flat markets (no trend), wait for clearer signal
    
    return signals

</think>