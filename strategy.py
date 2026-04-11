#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot with 1d volume confirmation and trend filter
# Long when price breaks above R4 (resistance 4) with volume > 1.5x 20-period average and 1d trend up
# Short when price breaks below S4 (support 4) with volume > 1.5x 20-period average and 1d trend down
# Exit when price returns to previous period's close or trend reverses
# Camarilla levels from 1d: R4 = close + 1.1*(high-low), S4 = close - 1.1*(high-low)
# Designed for 12-37 trades/year on 6h timeframe with strong trend following and low turnover

name = "6h_1d_camarilla_volume_trend_v1"
timeframe = "6h"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d high, low, close for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R4 and S4
    # R4 = close + 1.1*(high-low), S4 = close - 1.1*(high-low)
    camarilla_r4 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s4 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align Camarilla levels to 6h timeframe (wait for 1d bar to close)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period average volume for volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: price relative to 1d EMA50
        is_uptrend = close[i] > ema_50_1d_aligned[i]
        is_downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions: breakout of Camarilla R4/S4
        camarilla_breakout_up = close[i] > camarilla_r4_aligned[i-1]  # Break above previous period's R4
        camarilla_breakdown_down = close[i] < camarilla_s4_aligned[i-1]  # Break below previous period's S4
        
        long_entry = camarilla_breakout_up and volume_filter and is_uptrend
        short_entry = camarilla_breakdown_down and volume_filter and is_downtrend
        
        # Exit conditions: return to previous period's close or trend reversal
        long_exit = (close[i] < close_1d[i-1]) or (not is_uptrend)  # Return to prev 1d close or trend change
        short_exit = (close[i] > close_1d[i-1]) or (not is_downtrend)  # Return to prev 1d close or trend change
        
        # Priority: entry > exit > hold
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
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals