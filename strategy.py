#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1w weekly pivot direction + volume confirmation
# - Long when price breaks above 6h Donchian(20) high AND weekly pivot shows bullish bias (price > weekly pivot) AND volume > 1.5x 20-period average
# - Short when price breaks below 6h Donchian(20) low AND weekly pivot shows bearish bias (price < weekly pivot) AND volume > 1.5x 20-period average
# - Exit when price crosses 6h Donchian(20) midline
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Weekly pivot provides structural bias from higher timeframe (1w) to filter breakouts
# - Volume confirmation reduces false breakouts
# - Donchian breakouts capture strong momentum moves in both bull and bear markets

name = "6h_1w_donchian_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Pre-compute 6h Donchian channels (20)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian high/low (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Pre-compute 6h volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1w pivot points (using prior week's OHLC)
    # Weekly pivot = (Prior week HIGH + LOW + CLOSE) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    
    # Weekly bias: bullish if price > weekly pivot, bearish if price < weekly pivot
    weekly_bullish = weekly_pivot > 0  # placeholder, will be replaced with actual comparison after alignment
    weekly_bearish = weekly_pivot > 0  # placeholder
    
    # Align HTF indicators to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    # For bias, we need to compare close vs weekly pivot after alignment
    # We'll compute this inside the loop using aligned values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(weekly_pivot_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Determine weekly bias based on current close vs aligned weekly pivot
        weekly_bias_bullish = close[i] > weekly_pivot_aligned[i]
        weekly_bias_bearish = close[i] < weekly_pivot_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Donchian high AND weekly bias bullish AND volume spike
            if (close[i] > donch_high[i] and 
                weekly_bias_bullish and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Donchian low AND weekly bias bearish AND volume spike
            elif (close[i] < donch_low[i] and 
                  weekly_bias_bearish and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit condition: price crosses Donchian midline
            if (position == 1 and close[i] < donch_mid[i]) or \
               (position == -1 and close[i] > donch_mid[i]):
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1w weekly pivot direction + volume confirmation
# - Long when price breaks above 6h Donchian(20) high AND weekly pivot shows bullish bias (price > weekly pivot) AND volume > 1.5x 20-period average
# - Short when price breaks below 6h Donchian(20) low AND weekly pivot shows bearish bias (price < weekly pivot) AND volume > 1.5x 20-period average
# - Exit when price crosses 6h Donchian(20) midline
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Weekly pivot provides structural bias from higher timeframe (1w) to filter breakouts
# - Volume confirmation reduces false breakouts
# - Donchian breakouts capture strong momentum moves in both bull and bear markets

name = "6h_1w_donchian_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Pre-compute 6h Donchian channels (20)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian high/low (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Pre-compute 6h volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1w pivot points (using prior week's OHLC)
    # Weekly pivot = (Prior week HIGH + LOW + CLOSE) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    
    # Align HTF indicators to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(weekly_pivot_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Determine weekly bias based on current close vs aligned weekly pivot
        weekly_bias_bullish = close[i] > weekly_pivot_aligned[i]
        weekly_bias_bearish = close[i] < weekly_pivot_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Donchian high AND weekly bias bullish AND volume spike
            if (close[i] > donch_high[i] and 
                weekly_bias_bullish and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Donchian low AND weekly bias bearish AND volume spike
            elif (close[i] < donch_low[i] and 
                  weekly_bias_bearish and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit condition: price crosses Donchian midline
            if (position == 1 and close[i] < donch_mid[i]) or \
               (position == -1 and close[i] > donch_mid[i]):
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals