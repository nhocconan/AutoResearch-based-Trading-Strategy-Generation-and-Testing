#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike_ATRStop_v1
Hypothesis: On 1d timeframe, trade Camarilla R1/S1 breakouts with 1w EMA34 trend filter and volume spike confirmation. ATR-based stoploss limits drawdown. Target 30-100 total trades over 4 years by requiring confluence of weekly trend, volume confirmation, and price structure breakout. Designed to work in both bull and bear markets via trend filter and avoiding low-volume false breakouts.
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
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla R1 and S1 from previous day
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_r1 = df_1d['close'].values + (1.1 * (df_1d['high'].values - df_1d['low'].values) / 12)
    camarilla_s1 = df_1d['close'].values - (1.1 * (df_1d['high'].values - df_1d['low'].values) / 12)
    
    # Shift by 1 to use previous day's levels
    camarilla_r1 = np.roll(camarilla_r1, 1)
    camarilla_s1 = np.roll(camarilla_s1, 1)
    camarilla_r1[0] = np.nan
    camarilla_s1[0] = np.nan
    
    # Get 1w data for HTF trend (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # 1w EMA(34) for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # ATR(14) for stoploss
    tr1 = pd.Series(high).rolling(window=2).max() - pd.Series(low).rolling(window=2).min()
    tr2 = abs(pd.Series(high).rolling(window=2).max() - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low).rolling(window=2).min() - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Align HTF indicators to 1d timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    # Warmup: max of EMA(34) 1w (need 34 bars), volume MA (20), ATR(14)
    start_idx = max(34, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_34_1w_val = ema_34_1w_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol_spike = volume_spike[i]
        camarilla_r1 = camarilla_r1_aligned[i]
        camarilla_s1 = camarilla_s1_aligned[i]
        atr_val = atr[i]
        
        # Trend filter: price > EMA34 (uptrend) or < EMA34 (downtrend)
        uptrend = close_val > ema_34_1w_val
        downtrend = close_val < ema_34_1w_val
        
        if position == 0:
            # Long: break above Camarilla R1 with uptrend and volume spike
            long_signal = (high_val > camarilla_r1) and \
                          uptrend and \
                          vol_spike
            
            # Short: break below Camarilla S1 with downtrend and volume spike
            short_signal = (low_val < camarilla_s1) and \
                           downtrend and \
                           vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                atr_at_entry = atr_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                atr_at_entry = atr_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: ATR-based stoploss or trend reversal
            if close_val < entry_price - 2.0 * atr_at_entry or close_val < ema_34_1w_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: ATR-based stoploss or trend reversal
            if close_val > entry_price + 2.0 * atr_at_entry or close_val > ema_34_1w_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike_ATRStop_v1"
timeframe = "1d"
leverage = 1.0