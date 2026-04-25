#!/usr/bin/env python3
"""
12h Camarilla R1/S1 Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Camarilla pivot levels (R1/S1) from daily timeframe act as intraday support/resistance.
Breakouts above R1 or below S1 with 1d EMA34 trend filter and volume spike capture momentum.
Works in bull/bear via trend filter and volatility-based position sizing.
Target: 12-37 trades/year on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_r1 = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low']) / 12
    camarilla_s1 = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low']) / 12
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1.values)
    
    # Calculate ATR(14) for stoploss and volatility filter
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for data to propagate
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_34 = ema_34_aligned[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        atr_val = atr[i]
        
        # Volume spike: current volume > 2.0 * 50-period average
        if i >= 50:
            vol_ma_50 = np.mean(volume[i-49:i+1])
        else:
            vol_ma_50 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_50
        
        # Trend filter
        uptrend = curr_close > ema_34
        downtrend = curr_close < ema_34
        
        if position == 0:
            # Long: price breaks above Camarilla R1 AND volume spike AND uptrend
            long_condition = (curr_high > r1) and volume_spike and uptrend
            # Short: price breaks below Camarilla S1 AND volume spike AND downtrend
            short_condition = (curr_low < s1) and volume_spike and downtrend
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.5*ATR below entry) or trend reversal
            if curr_close <= entry_price - 2.5 * atr_val or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.5*ATR above entry) or trend reversal
            if curr_close >= entry_price + 2.5 * atr_val or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0