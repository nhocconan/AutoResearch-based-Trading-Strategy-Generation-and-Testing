#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d Trend Filter and Volume Confirmation
# - Williams Alligator (13,8,5 SMAs) on 12h to identify trend direction
# - 1d EMA50 as trend filter to ensure alignment with higher timeframe trend
# - Volume spike (>2x 20-period average) to confirm momentum
# - Designed for 12h timeframe to target 50-150 total trades over 4 years
# - Works in bull/bear by requiring trend alignment across timeframes
# - Entry: Price crosses above/below Alligator jaws with trend and volume confirmation
# - Exit: Price crosses back across Alligator teeth or trend reversal

name = "12h_WilliamsAlligator_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams Alligator: Jaws (13), Teeth (8), Lips (5) SMAs
    # Using typical price (H+L+C)/3 for better representation
    typical_price_12h = (high_12h + low_12h + close_12h) / 3.0
    
    jaws_12h = pd.Series(typical_price_12h).rolling(window=13, min_periods=13).mean().values
    teeth_12h = pd.Series(typical_price_12h).rolling(window=8, min_periods=8).mean().values
    lips_12h = pd.Series(typical_price_12h).rolling(window=5, min_periods=5).mean().values
    
    # Align Alligator lines to 12h timeframe (no additional delay needed for SMAs)
    jaws_12h_aligned = align_htf_to_ltf(prices, df_12h, jaws_12h)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaws_12h_aligned[i]) or np.isnan(teeth_12h_aligned[i]) or 
            np.isnan(lips_12h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above jaws with lips > teeth (bullish alignment) 
            #        + 1d uptrend + volume spike
            bullish_alignment = lips_12h_aligned[i] > teeth_12h_aligned[i]
            bearish_alignment = lips_12h_aligned[i] < teeth_12h_aligned[i]
            
            long_cond = (close[i] > jaws_12h_aligned[i] and 
                        bullish_alignment and
                        ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and
                        volume_spike[i])
            
            # Short: price crosses below jaws with lips < teeth (bearish alignment)
            #        + 1d downtrend + volume spike
            short_cond = (close[i] < jaws_12h_aligned[i] and 
                         bearish_alignment and
                         ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below teeth or bearish alignment
            if (close[i] < teeth_12h_aligned[i] or 
                lips_12h_aligned[i] < teeth_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above teeth or bullish alignment
            if (close[i] > teeth_12h_aligned[i] or 
                lips_12h_aligned[i] > teeth_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals