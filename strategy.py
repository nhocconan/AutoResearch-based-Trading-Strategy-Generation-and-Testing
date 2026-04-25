#!/usr/bin/env python3
"""
12h Williams Alligator + 1d EMA34 Trend + Volume Spike
Hypothesis: Williams Alligator identifies trending vs ranging markets. In trending markets (jaws/lips/teeth aligned), 
we trade breakouts of the Alligator's teeth in the direction of the 1d EMA34 trend. Volume spike confirms participation.
Works in bull/bear by trend-filtering breakouts. Target: 12-37 trades/year (50-150 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator on 12h timeframe
    # Jaws (Balance Line): 13-period SMMA, shifted 8 bars ahead
    # Teeth (Balance Line): 8-period SMMA, shifted 5 bars ahead  
    # Lips (Balance Line): 5-period SMMA, shifted 3 bars ahead
    # SMMA = Smoothed Moving Average (similar to EMA but with different smoothing)
    # We'll use EMA as approximation for SMMA for simplicity and performance
    
    # Calculate 12h EMAs for Alligator components
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # Jaws: 13-period EMA of median price, shifted 8 bars
    median_price_12h = (df_12h['high'] + df_12h['low']) / 2.0
    jaws_12h = pd.Series(median_price_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    jaws_12h_shifted = np.roll(jaws_12h, 8)  # shift 8 bars ahead
    jaws_12h_shifted[:8] = np.nan  # fill shifted values with NaN
    
    # Teeth: 8-period EMA of median price, shifted 5 bars
    teeth_12h = pd.Series(median_price_12h).ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth_12h_shifted = np.roll(teeth_12h, 5)  # shift 5 bars ahead
    teeth_12h_shifted[:5] = np.nan
    
    # Lips: 5-period EMA of median price, shifted 3 bars
    lips_12h = pd.Series(median_price_12h).ewm(span=5, adjust=False, min_periods=5).mean().values
    lips_12h_shifted = np.roll(lips_12h, 3)  # shift 3 bars ahead
    lips_12h_shifted[:3] = np.nan
    
    # Align Alligator lines to lower timeframe (12h->12h is identity, but we do it for consistency)
    jaws_aligned = align_htf_to_ltf(prices, df_12h, jaws_12h_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_12h_shifted)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator warmup + EMA34
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_34_aligned[i]
        jaw_val = jaws_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Alligator alignment: check if trending (jaws, teeth, lips are ordered and separated)
        # In uptrend: lips > teeth > jaws
        # In downtrend: lips < teeth < jaws
        is_uptrend_aligned = lips_val > teeth_val > jaw_val
        is_downtrend_aligned = lips_val < teeth_val < jaw_val
        
        # Entry conditions: trade in direction of 1d EMA34 trend when Alligator confirms trend
        if position == 0:
            # Long: price above lips AND 1d EMA34 uptrend AND Alligator confirms uptrend
            long_condition = (curr_close > lips_val) and (curr_close > ema_trend) and is_uptrend_aligned and volume_spike
            # Short: price below lips AND 1d EMA34 downtrend AND Alligator confirms downtrend
            short_condition = (curr_close < lips_val) and (curr_close < ema_trend) and is_downtrend_aligned and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below teeth OR trend breaks
            if curr_close < teeth_val or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above teeth OR trend breaks
            if curr_close > teeth_val or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0