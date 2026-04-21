#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1
Hypothesis: Camarilla R1/S1 breakout with volume confirmation (>1.8x 20-bar MA) and ATR-based stoploss works on 1h timeframe for BTC and ETH in both bull and bear markets. Uses 4h and 1d timeframes for signal direction (proven pattern: tight entries, volume confirmation, price channel structure). Target: 60-150 total trades over 4 years = 15-37/year for 1h. Uses session filter (08-20 UTC) to reduce noise.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-compute session hours for filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels on 4h timeframe (primary trend filter)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h = np.roll(close_4h, 1)
    prev_high_4h[0] = high_4h[0]
    prev_low_4h[0] = low_4h[0]
    prev_close_4h[0] = close_4h[0]
    
    range_4h = prev_high_4h - prev_low_4h
    camarilla_r1_4h = prev_close_4h + range_4h * 1.1 / 12
    camarilla_s1_4h = prev_close_4h - range_4h * 1.1 / 12
    
    # Align Camarilla levels from 4h to 1h timeframe
    camarilla_r1_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1_4h)
    camarilla_s1_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1_4h)
    
    # Calculate 1d ATR for volatility regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_1d[0] = tr2_1d[0] = tr3_1d[0] = np.nan
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume filter: 20-period average on 1h timeframe
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    # 1h ATR for stoploss
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r1_4h_aligned[i]) or np.isnan(camarilla_s1_4h_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation (>1.8x average to reduce trades)
        volume_ok = volume > 1.8 * vol_ma[i]
        
        # Volatility regime filter: only trade when 1d ATR is above its 50-period MA (avoid choppy markets)
        atr_ma_50 = pd.Series(atr_1d_aligned).rolling(window=50, min_periods=50).mean().values
        vol_regime_ok = not np.isnan(atr_ma_50[i]) and atr_1d_aligned[i] > atr_ma_50[i]
        
        if position == 0 and in_session:
            # Long: price breaks above Camarilla R1 (4h) with volume and vol regime
            if price > camarilla_r1_4h_aligned[i]:
                if volume_ok and vol_regime_ok:
                    signals[i] = 0.20
                    position = 1
            # Short: price breaks below Camarilla S1 (4h) with volume and vol regime
            elif price < camarilla_s1_4h_aligned[i]:
                if volume_ok and vol_regime_ok:
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:
            # Exit: price closes below Camarilla S1 (4h) or ATR stoploss
            if price < camarilla_s1_4h_aligned[i] or price < prices['close'].iloc[i-1] - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: price closes above Camarilla R1 (4h) or ATR stoploss
            if price > camarilla_r1_4h_aligned[i] or price > prices['close'].iloc[i-1] + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1"
timeframe = "1h"
leverage = 1.0