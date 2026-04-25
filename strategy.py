#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSp
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike (>1.5x 20-bar avg). Enters long when price breaks above R1 in 1d uptrend, short when breaks below S1 in 1d downtrend. Uses discrete sizing (0.25) to limit fee churn. Designed for 4h timeframe with ~20-50 trades/year, works in bull/bear by following 1d trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Previous day's high, low, close for Camarilla calculation
    prev_high = np.roll(close_1d, 1)  # Using close as proxy for day's high/low (intraday not available)
    prev_low = np.roll(close_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = close_1d[0]
    prev_low[0] = close_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # Approximate daily range using 1d high/low if available, else use close-based proxy
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    daily_range = df_1d_high - df_1d_low
    r1 = prev_close + (daily_range * 1.1 / 12)
    s1 = prev_close - (daily_range * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need 1 day of data for Camarilla and EMA
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 in 1d uptrend with volume confirmation
            bullish_setup = (close[i] > r1_aligned[i]) and (close[i] > open_price[i]) and (close_1d[i] > ema_34_1d_aligned[i]) and volume_spike[i]
            # Short: price breaks below S1 in 1d downtrend with volume confirmation
            bearish_setup = (close[i] < s1_aligned[i]) and (close[i] < open_price[i]) and (close_1d[i] < ema_34_1d_aligned[i]) and volume_spike[i]
            
            if bullish_setup:
                signals[i] = 0.25
                position = 1
            elif bearish_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below S1 OR trend turns down
            if (close[i] < s1_aligned[i]) or (close_1d[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above R1 OR trend turns up
            if (close[i] > r1_aligned[i]) or (close_1d[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSp"
timeframe = "4h"
leverage = 1.0