#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_Pivot_Volume_Trend"
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
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Based on previous day's OHLC
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels
    # R4 = Close + ((High - Low) * 1.5000)
    # R3 = Close + ((High - Low) * 1.2500)
    # R2 = Close + ((High - Low) * 1.1666)
    # R1 = Close + ((High - Low) * 1.0833)
    # S1 = Close - ((High - Low) * 1.0833)
    # S2 = Close - ((High - Low) * 1.1666)
    # S3 = Close - ((High - Low) * 1.2500)
    # S4 = Close - ((High - Low) * 1.5000)
    
    high_low_diff = prev_high - prev_low
    r1 = prev_close + (high_low_diff * 1.0833)
    s1 = prev_close - (high_low_diff * 1.0833)
    r3 = prev_close + (high_low_diff * 1.2500)
    s3 = prev_close - (high_low_diff * 1.2500)
    
    # Align pivot levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation - 4-period average volume (2 days for 12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: Price crosses above S1 with volume confirmation
            if (price > s1_aligned[i] and 
                close[i-1] <= s1_aligned[i-1] and  # crossed above S1
                vol_ratio[i] > 1.3):
                signals[i] = 0.25
                position = 1
            # Short: Price crosses below R1 with volume confirmation
            elif (price < r1_aligned[i] and 
                  close[i-1] >= r1_aligned[i-1] and  # crossed below R1
                  vol_ratio[i] > 1.3):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below S3 or R1 (reversal)
            if (price < s3_aligned[i] or 
                price > r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above R3 or S1 (reversal)
            if (price > r3_aligned[i] or 
                price < s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_Pivot_Volume_Trend"
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
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Based on previous day's OHLC
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels
    # R4 = Close + ((High - Low) * 1.5000)
    # R3 = Close + ((High - Low) * 1.2500)
    # R2 = Close + ((High - Low) * 1.1666)
    # R1 = Close + ((High - Low) * 1.0833)
    # S1 = Close - ((High - Low) * 1.0833)
    # S2 = Close - ((High - Low) * 1.1666)
    # S3 = Close - ((High - Low) * 1.2500)
    # S4 = Close - ((High - Low) * 1.5000)
    
    high_low_diff = prev_high - prev_low
    r1 = prev_close + (high_low_diff * 1.0833)
    s1 = prev_close - (high_low_diff * 1.0833)
    r3 = prev_close + (high_low_diff * 1.2500)
    s3 = prev_close - (high_low_diff * 1.2500)
    
    # Align pivot levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation - 4-period average volume (2 days for 12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: Price crosses above S1 with volume confirmation
            if (price > s1_aligned[i] and 
                close[i-1] <= s1_aligned[i-1] and  # crossed above S1
                vol_ratio[i] > 1.3):
                signals[i] = 0.25
                position = 1
            # Short: Price crosses below R1 with volume confirmation
            elif (price < r1_aligned[i] and 
                  close[i-1] >= r1_aligned[i-1] and  # crossed below R1
                  vol_ratio[i] > 1.3):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below S3 or R1 (reversal)
            if (price < s3_aligned[i] or 
                price > r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above R3 or S1 (reversal)
            if (price > r3_aligned[i] or 
                price < s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals