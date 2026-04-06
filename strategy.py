#!/usr/bin/env python3
"""
4h Volume-Weighted Moving Average Crossover with 12h Trend Filter and Volume Confirmation v1
Hypothesis: VWMA crossover captures momentum shifts; 12h trend filter avoids counter-trend trades; volume confirmation ensures institutional participation. Designed for 75-200 trades over 4 years to minimize fee drag while adapting to bull/bear markets via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_vwma_crossover_12h_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # 12h VWMA for trend filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Typical price for VWMA
    typical_price_12h = (high_12h + low_12h + close_12h) / 3
    
    # VWMA(20) on 12h
    vwma_num = np.convolve(typical_price_12h * volume_12h, np.ones(20), 'full')[:len(typical_price_12h)]
    vwma_den = np.convolve(volume_12h, np.ones(20), 'full')[:len(volume_12h)]
    vwma_12h = np.divide(vwma_num, vwma_den, out=np.full_like(vwma_num, np.nan), where=vwma_den!=0)
    
    # VWMA(50) on 12h
    vwma_num50 = np.convolve(typical_price_12h * volume_12h, np.ones(50), 'full')[:len(typical_price_12h)]
    vwma_den50 = np.convolve(volume_12h, np.ones(50), 'full')[:len(volume_12h)]
    vwma_50_12h = np.divide(vwma_num50, vwma_den50, out=np.full_like(vwma_num50, np.nan), where=vwma_den50!=0)
    
    # Trend: 1 if fast VWMA > slow VWMA, -1 otherwise
    trend_12h = np.where(vwma_12h > vwma_50_12h, 1, -1)
    trend_12h = np.where(np.isnan(vwma_12h) | np.isnan(vwma_50_12h), 0, trend_12h)
    
    # Align trend to 4h
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Typical price for VWMA
    typical_price = (high + low + close) / 3
    
    # VWMA(20) on 4h
    vwma_num_fast = np.convolve(typical_price * volume, np.ones(20), 'full')[:len(typical_price)]
    vwma_den_fast = np.convolve(volume, np.ones(20), 'full')[:len(volume)]
    vwma_fast = np.divide(vwma_num_fast, vwma_den_fast, out=np.full_like(vwma_num_fast, np.nan), where=vwma_den_fast!=0)
    
    # VWMA(50) on 4h
    vwma_num_slow = np.convolve(typical_price * volume, np.ones(50), 'full')[:len(typical_price)]
    vwma_den_slow = np.convolve(volume, np.ones(50), 'full')[:len(volume)]
    vwma_slow = np.divide(vwma_num_slow, vwma_den_slow, out=np.full_like(vwma_num_slow, np.nan), where=vwma_den_slow!=0)
    
    # Volume filter
    vol_ma = np.convolve(volume, np.ones(20), 'full')[:len(volume)] / 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 20)  # For VWMA and volume
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(vwma_fast[i]) or np.isnan(vwma_slow[i]) or
            np.isnan(trend_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: VWMA cross opposite direction OR trend change
        if position == 1:  # long position
            # Exit: VWMA cross down OR trend turns bearish
            if (vwma_fast[i] <= vwma_slow[i] and vwma_fast[i-1] > vwma_slow[i-1]) or trend_12h_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: VWMA cross up OR trend turns bullish
            if (vwma_fast[i] >= vwma_slow[i] and vwma_fast[i-1] < vwma_slow[i-1]) or trend_12h_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: VWMA cross + trend alignment + volume
            bull_cross = vwma_fast[i] > vwma_slow[i] and vwma_fast[i-1] <= vwma_slow[i-1]
            bear_cross = vwma_fast[i] < vwma_slow[i] and vwma_fast[i-1] >= vwma_slow[i-1]
            
            bull_entry = bull_cross and trend_12h_aligned[i] == 1 and volume[i] > vol_ma[i] * 1.5
            bear_entry = bear_cross and trend_12h_aligned[i] == -1 and volume[i] > vol_ma[i] * 1.5
            
            if bull_entry:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_entry:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
4h Volume-Weighted Moving Average Crossover with 12h Trend Filter and Volume Confirmation v1
Hypothesis: VWMA crossover captures momentum shifts; 12h trend filter avoids counter-trend trades; volume confirmation ensures institutional participation. Designed for 75-200 trades over 4 years to minimize fee drag while adapting to bull/bear markets via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_vwma_crossover_12h_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # 12h VWMA for trend filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Typical price for VWMA
    typical_price_12h = (high_12h + low_12h + close_12h) / 3
    
    # VWMA(20) on 12h
    vwma_num = np.convolve(typical_price_12h * volume_12h, np.ones(20), 'full')[:len(typical_price_12h)]
    vwma_den = np.convolve(volume_12h, np.ones(20), 'full')[:len(volume_12h)]
    vwma_12h = np.divide(vwma_num, vwma_den, out=np.full_like(vwma_num, np.nan), where=vwma_den!=0)
    
    # VWMA(50) on 12h
    vwma_num50 = np.convolve(typical_price_12h * volume_12h, np.ones(50), 'full')[:len(typical_price_12h)]
    vwma_den50 = np.convolve(volume_12h, np.ones(50), 'full')[:len(volume_12h)]
    vwma_50_12h = np.divide(vwma_num50, vwma_den50, out=np.full_like(vwma_num50, np.nan), where=vwma_den50!=0)
    
    # Trend: 1 if fast VWMA > slow VWMA, -1 otherwise
    trend_12h = np.where(vwma_12h > vwma_50_12h, 1, -1)
    trend_12h = np.where(np.isnan(vwma_12h) | np.isnan(vwma_50_12h), 0, trend_12h)
    
    # Align trend to 4h
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Typical price for VWMA
    typical_price = (high + low + close) / 3
    
    # VWMA(20) on 4h
    vwma_num_fast = np.convolve(typical_price * volume, np.ones(20), 'full')[:len(typical_price)]
    vwma_den_fast = np.convolve(volume, np.ones(20), 'full')[:len(volume)]
    vwma_fast = np.divide(vwma_num_fast, vwma_den_fast, out=np.full_like(vwma_num_fast, np.nan), where=vwma_den_fast!=0)
    
    # VWMA(50) on 4h
    vwma_num_slow = np.convolve(typical_price * volume, np.ones(50), 'full')[:len(typical_price)]
    vwma_den_slow = np.convolve(volume, np.ones(50), 'full')[:len(volume)]
    vwma_slow = np.divide(vwma_num_slow, vwma_den_slow, out=np.full_like(vwma_num_slow, np.nan), where=vwma_den_slow!=0)
    
    # Volume filter
    vol_ma = np.convolve(volume, np.ones(20), 'full')[:len(volume)] / 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 20)  # For VWMA and volume
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(vwma_fast[i]) or np.isnan(vwma_slow[i]) or
            np.isnan(trend_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: VWMA cross opposite direction OR trend change
        if position == 1:  # long position
            # Exit: VWMA cross down OR trend turns bearish
            if (vwma_fast[i] <= vwma_slow[i] and vwma_fast[i-1] > vwma_slow[i-1]) or trend_12h_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: VWMA cross up OR trend turns bullish
            if (vwma_fast[i] >= vwma_slow[i] and vwma_fast[i-1] < vwma_slow[i-1]) or trend_12h_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: VWMA cross + trend alignment + volume
            bull_cross = vwma_fast[i] > vwma_slow[i] and vwma_fast[i-1] <= vwma_slow[i-1]
            bear_cross = vwma_fast[i] < vwma_slow[i] and vwma_fast[i-1] >= vwma_slow[i-1]
            
            bull_entry = bull_cross and trend_12h_aligned[i] == 1 and volume[i] > vol_ma[i] * 1.5
            bear_entry = bear_cross and trend_12h_aligned[i] == -1 and volume[i] > vol_ma[i] * 1.5
            
            if bull_entry:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_entry:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals