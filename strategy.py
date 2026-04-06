#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams %R with Daily EMA200 Filter and Volume Spike.
# Williams %R identifies overbought/oversold conditions. 
# In strong trends (price > EMA200 for long, < EMA200 for short), we take counter-trend entries at extremes.
# Volume spike (>2x 20-period average) confirms momentum exhaustion.
# Works in both bull and bear markets via EMA200 trend filter.
# Target: 50-150 trades over 4 years (12-37/year).

name = "6h_williamsr_daily_ema200_vol_v3"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on daily closes
    ema200_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        ema200_1d[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            ema200_1d[i] = (close_1d[i] * 2 + ema200_1d[i-1] * 198) / 200
    
    # Align daily EMA200 to 6h timeframe (shifted by 1 daily bar)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Williams %R (14-period) on 6h data
    willr = np.full(n, np.nan)
    for i in range(13, n):
        highest_high = np.max(high[i-13:i+1])
        lowest_low = np.min(low[i-13:i+1])
        if highest_high != lowest_low:
            willr[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    # Volume filter: current volume > 2x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if Williams %R or EMA not available
        if np.isnan(willr[i]) or np.isnan(ema200_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 2.0
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Williams %R returns above -20 (overbought) or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.5 * atr_approx
            
            if (willr[i] >= -20 or close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Williams %R returns below -80 (oversold) or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.5 * atr_approx
            
            if (willr[i] <= -80 or close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend filter
            if volume_filter:
                # In uptrend (price > EMA200), look for oversold bounces
                if (close[i] > ema200_aligned[i] and 
                    willr[i] <= -80 and 
                    willr[i-1] > -80):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # In downtrend (price < EMA200), look for overbought bounces
                elif (close[i] < ema200_aligned[i] and 
                      willr[i] >= -20 and 
                      willr[i-1] < -20):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams %R with Daily EMA200 Filter and Volume Spike.
# Williams %R identifies overbought/oversold conditions. 
# In strong trends (price > EMA200 for long, < EMA200 for short), we take counter-trend entries at extremes.
# Volume spike (>2x 20-period average) confirms momentum exhaustion.
# Works in both bull and bear markets via EMA200 trend filter.
# Target: 50-150 trades over 4 years (12-37/year).

name = "6h_williamsr_daily_ema200_vol_v3"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on daily closes
    ema200_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        ema200_1d[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            ema200_1d[i] = (close_1d[i] * 2 + ema200_1d[i-1] * 198) / 200
    
    # Align daily EMA200 to 6h timeframe (shifted by 1 daily bar)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Williams %R (14-period) on 6h data
    willr = np.full(n, np.nan)
    for i in range(13, n):
        highest_high = np.max(high[i-13:i+1])
        lowest_low = np.min(low[i-13:i+1])
        if highest_high != lowest_low:
            willr[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    # Volume filter: current volume > 2x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if Williams %R or EMA not available
        if np.isnan(willr[i]) or np.isnan(ema200_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 2.0
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Williams %R returns above -20 (overbought) or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.5 * atr_approx
            
            if (willr[i] >= -20 or close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Williams %R returns below -80 (oversold) or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.5 * atr_approx
            
            if (willr[i] <= -80 or close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend filter
            if volume_filter:
                # In uptrend (price > EMA200), look for oversold bounces
                if (close[i] > ema200_aligned[i] and 
                    willr[i] <= -80 and 
                    willr[i-1] > -80):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # In downtrend (price < EMA200), look for overbought bounces
                elif (close[i] < ema200_aligned[i] and 
                      willr[i] >= -20 and 
                      willr[i-1] < -20):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals