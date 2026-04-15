#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R1/S1 breakout with volume confirmation and ATR filter
# Uses Camarilla pivot levels from 1d timeframe: break above R1 or below S1 with volume spike
# and elevated volatility (ATR > 1.5% of price) to filter low-quality breakouts.
# Works in both bull/bear markets by capturing momentum bursts after pivot level tests.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Position size: 0.25 (25% of capital) to balance return and drawdown.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (R1, S1)
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1/12
    # S1 = C - (H - L) * 1.1/12
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_1d = df_1d['high'] - df_1d['low']
    camarilla_r1_1d = df_1d['close'] + range_1d * 1.1 / 12
    camarilla_s1_1d = df_1d['close'] - range_1d * 1.1 / 12
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d.values)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d.values)
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            volume_filter = volume[i] > 1.5 * vol_ma
        else:
            volume_filter = False
        
        # Volatility filter: ATR > 1.5% of price
        vol_filter = atr_14_1d_aligned[i] > 0.015 * close[i]
        
        # Long: break above R1 with volume and volatility confirmation
        if (close[i] > camarilla_r1_aligned[i] and 
            volume_filter and 
            vol_filter):
            signals[i] = 0.25
            
        # Short: break below S1 with volume and volatility confirmation
        elif (close[i] < camarilla_s1_aligned[i] and 
              volume_filter and 
              vol_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_R1S1_Breakout_VolATR_v1"
timeframe = "6h"
leverage = 1.0