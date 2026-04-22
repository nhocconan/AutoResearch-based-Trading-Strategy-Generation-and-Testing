#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA trend filter and volume confirmation
# Long when price breaks above R1 + price > 1d EMA34 + volume spike
# Short when price breaks below S1 + price < 1d EMA34 + volume spike
# Exit when price crosses back through Pivot Point or trend reverses
# Designed for low trade frequency (~20-40/year) to minimize fee drain and work in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 34-period EMA on 1d close for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d candle
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # Pivot Point (PP) = (H+L+C)/3
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    camarilla_pp = (high_1d + low_1d + close_1d) / 3
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(pp_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        pp = pp_aligned[i]
        ema_val = ema_34_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above R1 + uptrend + volume spike
            if price > r1 and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S1 + downtrend + volume spike
            elif price < s1 and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses back through Pivot Point or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below PP or trend turns down
                if price < pp or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above PP or trend turns up
                if price > pp or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0