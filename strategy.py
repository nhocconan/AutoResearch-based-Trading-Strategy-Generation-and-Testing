#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d/200 EMA trend filter and volume spike
# Long when Jaw < Teeth < Lips (bullish alignment) + close > 1d EMA200 + volume spike
# Short when Jaw > Teeth > Lips (bearish alignment) + close < 1d EMA200 + volume spike
# Exit when Alligator alignment breaks or trend reverses
# Williams Alligator uses SMAs of median price (HL/2) with specific periods (13,8,5) and shifts (8,5,3)
# Designed for low trade frequency (~15-30/year on 12h) to minimize fee drain.
# Trend-following nature works in bull markets; volatility filter helps in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 200-period EMA on 1d close for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate Williams Alligator on 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Median price = (High + Low) / 2
    median_price = (high + low) / 2
    
    # Alligator lines: SMAs of median price with specific periods and shifts
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate 20-period average volume for volume spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Alligator alignment conditions
        bullish_alignment = jaw[i] < teeth[i] < lips[i]
        bearish_alignment = jaw[i] > teeth[i] > lips[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: bullish alignment + uptrend + volume spike
            if bullish_alignment and price > ema_200_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish alignment + downtrend + volume spike
            elif bearish_alignment and price < ema_200_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: Alligator alignment breaks or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when bullish alignment breaks or trend turns down
                if not bullish_alignment or price < ema_200_aligned[i]:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when bearish alignment breaks or trend turns up
                if not bearish_alignment or price > ema_200_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA200_Volume"
timeframe = "12h"
leverage = 1.0