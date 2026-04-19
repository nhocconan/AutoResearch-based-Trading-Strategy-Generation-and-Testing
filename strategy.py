#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + Elder Ray + Volume Confirmation
# Uses 4h as primary timeframe with 1d trend filter (EMA34) and volume spike (>1.8x average)
# Long when: price > Alligator teeth (middle line), Bull Power > 0, volume confirmed
# Short when: price < Alligator teeth, Bear Power < 0, volume confirmed
# Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) SMAs of median price
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Volume confirmation ensures institutional participation in trends
# Target: 20-50 trades/year per symbol (~80-200 total over 4 years)

name = "4h_WilliamsAlligator_ElderRay_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator on 4h data (median price)
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)  # Shift forward by 8 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)  # Shift forward by 5 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)  # Shift forward by 3 bars
    
    # Calculate Elder Ray components
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean()
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13)  # Need volume MA and EMA data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw.iloc[i]) or np.isnan(teeth.iloc[i]) or 
            np.isnan(lips.iloc[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        teeth_val = teeth.iloc[i]
        lips_val = lips.iloc[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        ema_trend = ema_34_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.8 * vol_ma
        
        if position == 0:
            # Enter long: price > teeth AND Bull Power > 0 AND above 1d EMA34
            if price > teeth_val and bull_val > 0 and price > ema_trend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short: price < teeth AND Bear Power < 0 AND below 1d EMA34
            elif price < teeth_val and bear_val < 0 and price < ema_trend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price < lips OR Bull Power <= 0 OR below 1d EMA34
            if price < lips_val or bull_val <= 0 or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price > lips OR Bear Power >= 0 OR above 1d EMA34
            if price > lips_val or bear_val >= 0 or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals