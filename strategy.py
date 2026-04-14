#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy combining weekly Bollinger Band squeeze with daily volume breakout.
# Long when price breaks above upper Bollinger Band after a squeeze (low volatility) with volume confirmation.
# Short when price breaks below lower Bollinger Band after a squeeze with volume confirmation.
# Uses weekly Bollinger Bands to identify low-volatility regimes and daily breakouts for entry.
# Bollinger Band squeeze defined as BB width < 20th percentile of past 50 weeks.
# Exit when price returns to weekly middle band (mean reversion within the band).
# Designed to capture volatility expansion periods in both bull and bear markets.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Bollinger Bands (20, 2) on weekly
    bb_period = 20
    bb_std = 2
    
    # Middle band (SMA)
    middle_bb = pd.Series(close_1w).rolling(window=bb_period, min_periods=bb_period).mean().values
    
    # Standard deviation
    std_dev = pd.Series(close_1w).rolling(window=bb_period, min_periods=bb_period).std().values
    
    # Upper and lower bands
    upper_bb = middle_bb + (bb_std * std_dev)
    lower_bb = middle_bb - (bb_std * std_dev)
    
    # Bollinger Band width
    bb_width = (upper_bb - lower_bb) / middle_bb
    
    # Squeeze condition: BB width < 20th percentile of past 50 weeks
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=50).quantile(0.20).values
    squeeze_condition = bb_width < bb_width_percentile
    
    # Daily volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly indicators to daily timeframe
    middle_bb_aligned = align_htf_to_ltf(prices, df_1w, middle_bb)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1w, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1w, lower_bb)
    squeeze_aligned = align_htf_to_ltf(prices, df_1w, squeeze_condition)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(bb_period, 50)  # Need BB and percentile
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(middle_bb_aligned[i]) or 
            np.isnan(upper_bb_aligned[i]) or
            np.isnan(lower_bb_aligned[i]) or
            np.isnan(squeeze_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for breakouts after squeeze
            # Long: price breaks above upper BB after squeeze
            if (close[i] > upper_bb_aligned[i] and 
                squeeze_aligned[i] and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower BB after squeeze
            elif (close[i] < lower_bb_aligned[i] and 
                  squeeze_aligned[i] and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle BB
            if close[i] <= middle_bb_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to middle BB
            if close[i] >= middle_bb_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_BollingerSqueeze_VolumeBreakout_v1"
timeframe = "1d"
leverage = 1.0