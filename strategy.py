#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d VWAP filter and volume spike
# Long when Williams %R < -80 (oversold) + price > 1d VWAP + volume spike
# Short when Williams %R > -20 (overbought) + price < 1d VWAP + volume spike
# Exit when Williams %R returns to -50 (neutral) or trend reverses
# Designed for low trade frequency (~15-30/year) with edge in ranging markets
# Williams %R identifies exhaustion; VWAP filters trend direction; volume confirms strength

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 14-period Williams %R on 12h data (using close, high, low)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_12h) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Calculate 1d VWAP (typical price * volume cumulative / volume cumulative)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    vwap_num = np.cumsum(typical_price_1d * volume_1d)
    vwap_den = np.cumsum(volume_1d)
    vwap_1d = np.where(vwap_den != 0, vwap_num / vwap_den, typical_price_1d)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or 
            np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        wr = williams_r[i]
        vwap = vwap_1d_aligned[i]
        
        # Volume filter: current volume > 1.8 * 20-day average
        vol_spike = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long conditions: Williams %R oversold + price above VWAP + volume spike
            if wr < -80 and price > vwap and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R overbought + price below VWAP + volume spike
            elif wr > -20 and price < vwap and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: Williams %R returns to neutral (-50) or price crosses VWAP
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Williams %R returns to neutral or price drops below VWAP
                if wr >= -50 or price < vwap:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Williams %R returns to neutral or price rises above VWAP
                if wr <= -50 or price > vwap:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsR_VWAP_Volume"
timeframe = "12h"
leverage = 1.0