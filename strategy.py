#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hEMA34_Trend_VolumeSpike
Hypothesis: Camarilla R1/S1 levels from 12h act as intraday support/resistance. A break above R1 with volume spike and 12h uptrend signals long; break below S1 with volume spike and 12h downtrend signals short. Uses discrete position sizing (0.30) to limit fee drag. Works in both bull and bear markets by capturing breakouts from institutional levels with trend and volume confirmation. Target: 19-50 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for Camarilla levels (loaded ONCE)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Camarilla levels from previous 12h bar
    # Camarilla: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    # Camarilla R1 and S1 from previous 12h bar
    camarilla_r1 = c_12h + (h_12h - l_12h) * 1.1 / 12
    camarilla_s1 = c_12h - (h_12h - l_12h) * 1.1 / 12
    
    # 12h EMA34 for trend filter (loaded ONCE)
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to LTF (4h)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # 4h volume spike: current volume > 2.0 * 20-period volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start index: need volume MA (20) + aligned HTF arrays
    start_idx = max(20, 0)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Long: price breaks above camarilla R1 with volume spike and 12h uptrend
            long_breakout = (curr_close > camarilla_r1_aligned[i]) and vol_spike[i] and (curr_close > ema_34_12h_aligned[i])
            # Short: price breaks below camarilla S1 with volume spike and 12h downtrend
            short_breakout = (curr_close < camarilla_s1_aligned[i]) and vol_spike[i] and (curr_close < ema_34_12h_aligned[i])
            
            if long_breakout:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            # Exit: price breaks below camarilla S1 OR trend turns down
            if (curr_close < camarilla_s1_aligned[i]) or (curr_close < ema_34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Exit: price breaks above camarilla R1 OR trend turns up
            if (curr_close > camarilla_r1_aligned[i]) or (curr_close > ema_34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0