#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
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
    
    # Get 1w data for trend filter and Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly high, low, close for Camarilla levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla R3 and S3 levels for each week
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    camarilla_r3 = close_1w + (high_1w - low_1w) * 1.1 / 2
    camarilla_s3 = close_1w - (high_1w - low_1w) * 1.1 / 2
    
    # Align weekly Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # Get 1d data for volume average
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # Current volume for confirmation
    vol_series = pd.Series(volume)
    vol_ma20_current = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma20_1d_aligned[i]) or np.isnan(vol_ma20_current[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20_current[i]
        
        if position == 0:
            # Long: Break above Camarilla R3 with volume and above weekly trend (price > weekly close)
            if close[i] > camarilla_r3_aligned[i] and vol_ok and close[i] > close_1w[-1] if len(close_1w) > 0 else False:
                # Check if we have valid weekly close for current week
                weekly_close_val = close_1w[-1] if len(close_1w) > 0 else np.nan
                if not np.isnan(weekly_close_val) and close[i] > weekly_close_val:
                    signals[i] = 0.25
                    position = 1
            # Short: Break below Camarilla S3 with volume and below weekly trend (price < weekly close)
            elif close[i] < camarilla_s3_aligned[i] and vol_ok and close[i] < close_1w[-1] if len(close_1w) > 0 else False:
                weekly_close_val = close_1w[-1] if len(close_1w) > 0 else np.nan
                if not np.isnan(weekly_close_val) and close[i] < weekly_close_val:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: Price falls below Camarilla S3 or weekly trend reversal
            if close[i] < camarilla_s3_aligned[i] or close[i] < close_1w[-1] if len(close_1w) > 0 else False:
                weekly_close_val = close_1w[-1] if len(close_1w) > 0 else np.nan
                if not np.isnan(weekly_close_val) and (close[i] < camarilla_s3_aligned[i] or close[i] < weekly_close_val):
                    signals[i] = 0.0
                    position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above Camarilla R3 or weekly trend reversal
            if close[i] > camarilla_r3_aligned[i] or close[i] > close_1w[-1] if len(close_1w) > 0 else False:
                weekly_close_val = close_1w[-1] if len(close_1w) > 0 else np.nan
                if not np.isnan(weekly_close_val) and (close[i] > camarilla_r3_aligned[i] or close[i] > weekly_close_val):
                    signals[i] = 0.0
                    position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
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
    
    # Get 1w data for trend filter and Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly high, low, close for Camarilla levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla R3 and S3 levels for each week
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    camarilla_r3 = close_1w + (high_1w - low_1w) * 1.1 / 2
    camarilla_s3 = close_1w - (high_1w - low_1w) * 1.1 / 2
    
    # Align weekly Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # Get 1d data for volume average
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # Current volume for confirmation
    vol_series = pd.Series(volume)
    vol_ma20_current = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma20_1d_aligned[i]) or np.isnan(vol_ma20_current[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20_current[i]
        
        if position == 0:
            # Long: Break above Camarilla R3 with volume and above weekly trend (price > weekly close)
            if close[i] > camarilla_r3_aligned[i] and vol_ok and close[i] > close_1w[-1] if len(close_1w) > 0 else False:
                # Check if we have valid weekly close for current week
                weekly_close_val = close_1w[-1] if len(close_1w) > 0 else np.nan
                if not np.isnan(weekly_close_val) and close[i] > weekly_close_val:
                    signals[i] = 0.25
                    position = 1
            # Short: Break below Camarilla S3 with volume and below weekly trend (price < weekly close)
            elif close[i] < camarilla_s3_aligned[i] and vol_ok and close[i] < close_1w[-1] if len(close_1w) > 0 else False:
                weekly_close_val = close_1w[-1] if len(close_1w) > 0 else np.nan
                if not np.isnan(weekly_close_val) and close[i] < weekly_close_val:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: Price falls below Camarilla S3 or weekly trend reversal
            if close[i] < camarilla_s3_aligned[i] or close[i] < close_1w[-1] if len(close_1w) > 0 else False:
                weekly_close_val = close_1w[-1] if len(close_1w) > 0 else np.nan
                if not np.isnan(weekly_close_val) and (close[i] < camarilla_s3_aligned[i] or close[i] < weekly_close_val):
                    signals[i] = 0.0
                    position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above Camarilla R3 or weekly trend reversal
            if close[i] > camarilla_r3_aligned[i] or close[i] > close_1w[-1] if len(close_1w) > 0 else False:
                weekly_close_val = close_1w[-1] if len(close_1w) > 0 else np.nan
                if not np.isnan(weekly_close_val) and (close[i] > camarilla_r3_aligned[i] or close[i] > weekly_close_val):
                    signals[i] = 0.0
                    position = 0
            else:
                signals[i] = -0.25
    
    return signals