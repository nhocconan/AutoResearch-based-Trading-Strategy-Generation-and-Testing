#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator (13,8,5 SMAs) with 1d trend filter and volume confirmation
# Long when price > Jaw, Teeth > Lips (bullish alignment), 1d EMA34 rising, volume > 1.5x avg
# Short when price < Jaw, Teeth < Lips (bearish alignment), 1d EMA34 falling, volume > 1.5x avg
# Uses Alligator for trend identification, EMA34 for higher timeframe trend filter, volume for confirmation
# Targets 12-37 trades per year (48-148 over 4 years) for low fee drag and high win rate
# Works in both bull and bear markets due to dual timeframe trend alignment and volume confirmation

name = "12h_WilliamsAlligator_1dEMA34_Volume"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Williams Alligator: 13,8,5 period SMAs on median price
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # 13-period (blue line)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values   # 8-period (red line)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values    # 5-period (green line)
    
    # Calculate EMA34 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 13  # Need at least 13 periods for Jaw calculation
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        ema34_1d_val = ema34_1d_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: price > Jaw, Teeth > Lips (bullish alignment), 1d uptrend, volume confirmation
            if close_val > jaw_val and teeth_val > lips_val and ema34_1d_val > 0 and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: price < Jaw, Teeth < Lips (bearish alignment), 1d downtrend, volume confirmation
            elif close_val < jaw_val and teeth_val < lips_val and ema34_1d_val < 0 and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < Jaw OR Teeth < Lips (loss of bullish alignment) OR 1d trend down
            if close_val < jaw_val or teeth_val < lips_val or ema34_1d_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > Jaw OR Teeth > Lips (loss of bearish alignment) OR 1d trend up
            if close_val > jaw_val or teeth_val > lips_val or ema34_1d_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals