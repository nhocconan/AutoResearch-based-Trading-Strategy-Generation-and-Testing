#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Bollinger Band squeeze + 1d Camarilla pivot breakout
    # Long: BB squeeze (BW < 20th percentile) + price breaks above R3 Camarilla (1d)
    # Short: BB squeeze + price breaks below S3 Camarilla (1d)
    # Exit: BB expansion (BW > 50th percentile) OR price reverts to mean (close to BB middle)
    # Uses 6h for volatility squeeze detection, 1d for Camarilla pivot structure
    # Discrete position sizing (0.25) to minimize fee churn
    # Target: 50-150 total trades over 4 years (~12-37/year) to stay within limits
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 6h data for Bollinger Bands (call ONCE before loop)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivots (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 6h Bollinger Bands (20, 2)
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # BB middle (SMA20)
    sma20_6h = pd.Series(close_6h).rolling(window=20, min_periods=20).mean().values
    
    # BB standard deviation
    bb_std_6h = pd.Series(close_6h).rolling(window=20, min_periods=20).std().values
    
    # BB upper/lower bands
    bb_upper_6h = sma20_6h + 2 * bb_std_6h
    bb_lower_6h = sma20_6h - 2 * bb_std_6h
    
    # Bollinger Band Width (BW)
    bb_width_6h = (bb_upper_6h - bb_lower_6h) / sma20_6h
    
    # Align 6h Bollinger Band Width to 6h (no additional delay for price-based indicators)
    bb_width_aligned = align_htf_to_ltf(prices, df_6h, bb_width_6h)
    bb_middle_aligned = align_htf_to_ltf(prices, df_6h, sma20_6h)
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels (using previous day's OHLC)
    camarilla_h4 = np.full_like(close_1d, np.nan)
    camarilla_l4 = np.full_like(close_1d, np.nan)
    camarilla_h3 = np.full_like(close_1d, np.nan)
    camarilla_l3 = np.full_like(close_1d, np.nan)
    
    # Calculate for each day (starting from index 1 as we need previous day)
    for i in range(1, len(close_1d)):
        # Previous day's OHLC
        phigh = high_1d[i-1]
        plow = low_1d[i-1]
        pclose = close_1d[i-1]
        pivot = (phigh + plow + pclose) / 3
        range_ = phigh - plow
        
        # Camarilla levels
        camarilla_h4[i] = pclose + range_ * 1.1 / 2
        camarilla_l4[i] = pclose - range_ * 1.1 / 2
        camarilla_h3[i] = pclose + range_ * 1.1 / 4
        camarilla_l3[i] = pclose - range_ * 1.1 / 4
    
    # For first bar, use NaN (no previous day)
    camarilla_h4[0] = np.nan
    camarilla_l4[0] = np.nan
    camarilla_h3[0] = np.nan
    camarilla_l3[0] = np.nan
    
    # Align 1d Camarilla levels to 6h (wait for completed 1d bar)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Percentile lookback for BB width regime (50 bars ~ 6h*50 = 12.5 days)
    lookback = 50
    
    for i in range(lookback, n):
        # Skip if data not ready
        if (np.isnan(bb_width_aligned[i]) or np.isnan(bb_middle_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # BB squeeze detection: BB width below 20th percentile (low volatility)
        bb_width_slice = bb_width_aligned[i-lookback:i+1]
        valid_bw = bb_width_slice[~np.isnan(bb_width_slice)]
        if len(valid_bw) < 10:  # Need minimum valid data
            signals[i] = 0.0
            continue
            
        bw_percentile_20 = np.percentile(valid_bw, 20)
        bw_percentile_50 = np.percentile(valid_bw, 50)
        
        bb_squeeze = bb_width_aligned[i] < bw_percentile_20
        bb_expansion = bb_width_aligned[i] > bw_percentile_50
        
        # Price relative to BB middle
        price_to_middle = (close[i] - bb_middle_aligned[i]) / bb_middle_aligned[i]
        price_at_mean = np.abs(price_to_middle) < 0.005  # Within 0.5% of BB middle
        
        # Camarilla breakout conditions
        long_breakout = close[i] > h3_aligned[i]  # Break above H3
        short_breakout = close[i] < l3_aligned[i]  # Break below L3
        
        # Exit conditions
        long_exit = bb_expansion or price_at_mean
        short_exit = bb_expansion or price_at_mean
        
        # Entry logic: BB squeeze + Camarilla breakout
        long_entry = bb_squeeze and long_breakout
        short_entry = bb_squeeze and short_breakout
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_bb_squeeze_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0