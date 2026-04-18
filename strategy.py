#!/usr/bin/env python3
"""
4h_Pivot_R1S1_Breakout_VolumeSpike_12hEMA34_v2
Hypothesis: Camarilla R1/S1 breakout with volume spike and 12h EMA34 trend filter works in both bull and bear markets by capturing momentum with strict entry conditions. Reduced position size to 0.20 to lower drawdown and added ATR-based stoploss for risk control.
"""

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
    
    # Get 12h data for EMA34 trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close']
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Get 1d data for Camarilla pivot levels (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    close_1d = df_1d['close']
    
    # Calculate ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels for each day
    camarilla_range = (high_1d - low_1d)
    r1_level = close_1d + (1.1 * camarilla_range) / 12
    s1_level = close_1d - (1.1 * camarilla_range) / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_level)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_level)
    
    # Volume spike detection: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_12h_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        ema_trend = ema_34_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: break above R1 with 12h uptrend and volume spike
            if price > r1 and price > ema_trend and vol_spike:
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short: break below S1 with 12h downtrend and volume spike
            elif price < s1 and price < ema_trend and vol_spike:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        elif position == 1:
            signals[i] = 0.20
            # Exit: stoploss, or price returns to S1, or breaks below 12h EMA
            if price <= entry_price - 1.5 * atr[i] or price < s1 or price < ema_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.20
            # Exit: stoploss, or price returns to R1, or breaks above 12h EMA
            if price >= entry_price + 1.5 * atr[i] or price > r1 or price > ema_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Pivot_R1S1_Breakout_VolumeSpike_12hEMA34_v2"
timeframe = "4h"
leverage = 1.0