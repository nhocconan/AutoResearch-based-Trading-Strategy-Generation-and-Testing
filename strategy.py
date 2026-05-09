#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_ADX14_Trend_30Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for ADX and volume filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # ADX calculation (14-period)
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = abs(df_12h['high'] - df_12h['close'].shift(1))
    tr3 = abs(df_12h['low'] - df_12h['close'].shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    
    plus_dm = df_12h['high'].diff()
    minus_dm = df_12h['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = abs(minus_dm)
    
    atr = tr.rolling(window=14, min_periods=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=14, min_periods=14).mean()
    
    # Volume filter: current 12h volume > 30 * 30-period average (extreme spike)
    vol_series = pd.Series(df_12h['volume'].values)
    vol_ma = vol_series.rolling(window=30, min_periods=30).mean().values
    volume_filter_12h = df_12h['volume'].values > (vol_ma * 30)
    
    # Align all to 6h
    adx_6h = align_htf_to_ltf(prices, df_12h, adx.values)
    plus_di_6h = align_htf_to_ltf(prices, df_12h, plus_di.values)
    minus_di_6h = align_htf_to_ltf(prices, df_12h, minus_di.values)
    volume_filter_6h = align_htf_to_ltf(prices, df_12h, volume_filter_12h)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 42  # Need enough data for ADX calculation (14*3)
    
    for i in range(start_idx, n):
        if (np.isnan(adx_6h[i]) or np.isnan(plus_di_6h[i]) or 
            np.isnan(minus_di_6h[i]) or np.isnan(volume_filter_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_6h[i]
        plus_di_val = plus_di_6h[i]
        minus_di_val = minus_di_6h[i]
        vol_filter = volume_filter_6h[i]
        
        if position == 0:
            # Enter long: ADX > 25 (strong trend), +DI > -DI, volume spike
            if adx_val > 25 and plus_di_val > minus_di_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: ADX > 25, -DI > +DI, volume spike
            elif adx_val > 25 and minus_di_val > plus_di_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend weakens (ADX < 20) or DI crossover
            if adx_val < 20 or minus_di_val > plus_di_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend weakens or DI crossover
            if adx_val < 20 or plus_di_val > minus_di_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals