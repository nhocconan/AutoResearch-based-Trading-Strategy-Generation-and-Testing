#!/usr/bin/env python3
name = "1h_Camillo_R1S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (pre-compute)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h trend filter: EMA34
    df_4h = get_htf_data(prices, '4h')
    ema34_4h = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Daily volume average (for volume spike filter)
    df_1d = get_htf_data(prices, '1d')
    vol_avg_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Camarilla pivot levels (based on previous day)
    # Calculate from daily OHLC
    dm_high = df_1d['high'].values
    dm_low = df_1d['low'].values
    dm_close = df_1d['close'].values
    
    # Previous day's Camarilla levels
    range_d = dm_high - dm_low
    camarilla_h5 = dm_close + range_d * 1.1 / 2
    camarilla_h4 = dm_close + range_d * 1.1 / 4
    camarilla_h3 = dm_close + range_d * 1.1 / 6
    camarilla_l3 = dm_close - range_d * 1.1 / 6
    camarilla_l4 = dm_close - range_d * 1.1 / 4
    camarilla_l5 = dm_close - range_d * 1.1 / 2
    
    # Align to 1h
    h5 = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    h4 = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    h3 = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3 = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    l4 = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    l5 = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if outside session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if 4h trend or volume data not ready
        if np.isnan(ema34_4h_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume spike: current volume > 1.5x daily average
        vol_spike = volume[i] > (vol_avg_1d_aligned[i] * 1.5)
        
        if position == 0:
            # Long: Price > Camarilla H4 AND 4h uptrend AND volume spike
            if (close[i] > h4[i] and 
                close[i] > ema34_4h_aligned[i] and 
                vol_spike):
                signals[i] = 0.20
                position = 1
            # Short: Price < Camarilla L4 AND 4h downtrend AND volume spike
            elif (close[i] < l4[i] and 
                  close[i] < ema34_4h_aligned[i] and 
                  vol_spike):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Price < Camarilla L3 OR 4h trend turns down
            if (close[i] < l3[i] or close[i] < ema34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Price > Camarilla H3 OR 4h trend turns up
            if (close[i] > h3[i] or close[i] > ema34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals