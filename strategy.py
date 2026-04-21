#!/usr/bin/env python3
"""
1d_HTF_1w_Camarilla_R1S1_Breakout_VolumeSpike_ATRStop_V1
Hypothesis: Use weekly Camarilla R1/S1 levels from 1w HTF + 1d volume spike (>2x 20-bar MA) for breakout entry + ATR(14) stoploss (2.0x). 
Add regime filter: only trade when 1d ADX(14) > 20 (trending market filter) to reduce whipsaw. 
Weekly Camarilla provides stronger structural levels than daily, reducing false breakouts. 
Discrete position sizing (0.25) balances return and drawdown. Target 15-25 trades/year per symbol. 
Works in bull (breakouts capture momentum) and bear (tight stops limit losses during reversals, ADX filter avoids ranging markets).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')  # for weekly Camarilla levels
    
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # === Weekly Camarilla Levels (R1, S1) ===
    # Camarilla formula: based on previous week's range
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels for each week
    R1 = np.full_like(close_1w, np.nan)
    S1 = np.full_like(close_1w, np.nan)
    
    for i in range(1, len(close_1w)):
        # Previous week's OHLC
        phigh = high_1w[i-1]
        plow = low_1w[i-1]
        pclose = close_1w[i-1]
        
        # Camarilla R1 and S1
        R1[i] = pclose + (1.1/12) * (phigh - plow)
        S1[i] = pclose - (1.1/12) * (phigh - plow)
    
    # Align weekly levels to daily timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    
    # === 1d Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ADX (14-period) for regime filter
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = minus_dm[0] = 0
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum() / tr_sum
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum() / tr_sum
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(adx[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 2.0 * vol_ma[i]  # volume spike confirmation
        adx_ok = adx[i] > 20.0  # trending market filter: avoid ranging markets
        
        if position == 0:
            # Long: break above weekly R1 with volume spike and ADX > 20
            if price > R1_aligned[i-1] and vol_ok and adx_ok:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S1 with volume spike and ADX > 20
            elif price < S1_aligned[i-1] and vol_ok and adx_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: ATR stoploss or opposite signal
            if price < R1_aligned[i-1] - 2.0 * atr[i] or price < S1_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: ATR stoploss or opposite signal
            if price > S1_aligned[i-1] + 2.0 * atr[i] or price > R1_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_HTF_1w_Camarilla_R1S1_Breakout_VolumeSpike_ATRStop_V1"
timeframe = "1d"
leverage = 1.0