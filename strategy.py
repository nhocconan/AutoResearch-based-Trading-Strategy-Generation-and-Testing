#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1-day Bollinger Band squeeze and breakout
# Uses Bollinger Band width to identify low volatility periods (squeeze)
# Breaks out when price closes outside Bollinger Bands with volume confirmation
# Designed for 20-40 trades/year with focus on BTC/ETH performance

name = "4h_1d_bb_squeeze_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period SMA and standard deviation
    sma_20 = np.full_like(close_1d, np.nan)
    std_20 = np.full_like(close_1d, np.nan)
    
    for i in range(19, len(close_1d)):
        sma_20[i] = np.mean(close_1d[i-19:i+1])
        std_20[i] = np.std(close_1d[i-19:i+1])
    
    # Calculate upper and lower bands
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)
    bb_width = upper_band - lower_band
    
    # Calculate Bollinger Band width percentile (252-period for 1 year)
    bb_width_pct = np.full_like(bb_width, np.nan)
    for i in range(251, len(bb_width)):
        bb_width_pct[i] = scipy.stats.percentileofscore(bb_width[i-251:i+1], bb_width[i]) / 100.0
    
    # Identify squeeze: BB width below 20th percentile (low volatility)
    squeeze = bb_width_pct < 0.20
    
    # Align squeeze and bands to 4h
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze)
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # Calculate 20-period average volume for confirmation
    vol_1d = df_1d['volume'].values
    vol_avg_20 = np.full_like(vol_1d, np.nan)
    for i in range(19, len(vol_1d)):
        vol_avg_20[i] = np.mean(vol_1d[i-19:i+1])
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if any required data is invalid
        if (np.isnan(squeeze_aligned[i]) or 
            np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i]) or
            np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.3 * daily average volume
        vol_filter = volume[i] > 1.3 * vol_avg_aligned[i]
        
        # Breakout conditions
        breakout_long = (close[i] > upper_band_aligned[i] and vol_filter and squeeze_aligned[i])
        breakout_short = (close[i] < lower_band_aligned[i] and vol_filter and squeeze_aligned[i])
        
        # Exit when price returns to middle band (mean reversion)
        middle_band = (upper_band_aligned[i] + lower_band_aligned[i]) / 2
        exit_long = (position == 1 and close[i] <= middle_band)
        exit_short = (position == -1 and close[i] >= middle_band)
        
        # Priority: breakout > exit > hold
        if breakout_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Import scipy for percentile calculation
try:
    import scipy.stats
except ImportError:
    # Fallback method if scipy is not available
    def percentileofscore(scores, score):
        if len(scores) == 0:
            return 0.0
        return np.sum(scores <= score) / len(scores) * 100.0
    
    # Override the bb_width_pct calculation
    def generate_signals(prices):
        n = len(prices)
        if n < 50:
            return np.zeros(n)
        
        # Price arrays
        high = prices['high'].values
        low = prices['low'].values
        close = prices['close'].values
        volume = prices['volume'].values
        
        # Load daily data ONCE before loop
        df_1d = get_htf_data(prices, '1d')
        if len(df_1d) < 30:
            return np.zeros(n)
        
        # Calculate daily Bollinger Bands (20, 2)
        close_1d = df_1d['close'].values
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        
        # Calculate 20-period SMA and standard deviation
        sma_20 = np.full_like(close_1d, np.nan)
        std_20 = np.full_like(close_1d, np.nan)
        
        for i in range(19, len(close_1d)):
            sma_20[i] = np.mean(close_1d[i-19:i+1])
            std_20[i] = np.std(close_1d[i-19:i+1])
        
        # Calculate upper and lower bands
        upper_band = sma_20 + (2 * std_20)
        lower_band = sma_20 - (2 * std_20)
        bb_width = upper_band - lower_band
        
        # Calculate Bollinger Band width percentile (252-period for 1 year) - fallback method
        bb_width_pct = np.full_like(bb_width, np.nan)
        for i in range(251, len(bb_width)):
            # Calculate percentile manually: percentage of values <= current value
            past_values = bb_width[i-251:i+1]
            bb_width_pct[i] = np.sum(past_values <= bb_width[i]) / len(past_values)
        
        # Identify squeeze: BB width below 20th percentile (low volatility)
        squeeze = bb_width_pct < 0.20
        
        # Align squeeze and bands to 4h
        squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze)
        upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
        lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
        
        # Calculate 20-period average volume for confirmation
        vol_1d = df_1d['volume'].values
        vol_avg_20 = np.full_like(vol_1d, np.nan)
        for i in range(19, len(vol_1d)):
            vol_avg_20[i] = np.mean(vol_1d[i-19:i+1])
        vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
        
        signals = np.zeros(n)
        position = 0  # 1=long, -1=short, 0=flat
        
        for i in range(1, n):
            # Skip if any required data is invalid
            if (np.isnan(squeeze_aligned[i]) or 
                np.isnan(upper_band_aligned[i]) or 
                np.isnan(lower_band_aligned[i]) or
                np.isnan(vol_avg_aligned[i])):
                signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
                continue
            
            # Volume filter: current volume > 1.3 * daily average volume
            vol_filter = volume[i] > 1.3 * vol_avg_aligned[i]
            
            # Breakout conditions
            breakout_long = (close[i] > upper_band_aligned[i] and vol_filter and squeeze_aligned[i])
            breakout_short = (close[i] < lower_band_aligned[i] and vol_filter and squeeze_aligned[i])
            
            # Exit when price returns to middle band (mean reversion)
            middle_band = (upper_band_aligned[i] + lower_band_aligned[i]) / 2
            exit_long = (position == 1 and close[i] <= middle_band)
            exit_short = (position == -1 and close[i] >= middle_band)
            
            # Priority: breakout > exit > hold
            if breakout_long and position != 1:
                position = 1
                signals[i] = 0.25
            elif breakout_short and position != -1:
                position = -1
                signals[i] = -0.25
            elif position == 1 and exit_long:
                position = 0
                signals[i] = 0.0
            elif position == -1 and exit_short:
                position = 0
                signals[i] = 0.0
            else:
                # Hold current position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        
        return signals
except Exception as e:
    # If any import or calculation fails, return zero signals
    def generate_signals(prices):
        return np.zeros(len(prices))