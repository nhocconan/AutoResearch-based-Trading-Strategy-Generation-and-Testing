#!/usr/bin/env python3
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
    
    # Get 1d HTF data once before loop for daily ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility regime filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    # Align daily ATR to 12h timeframe
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 12h ATR(14) for position sizing and stoploss reference
    tr1_12h = high - low
    tr2_12h = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3_12h = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    atr_14_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 12h Donchian(20) channels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(atr_14_12h[i]) or np.isnan(atr_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: Only trade when volatility is elevated (ATR > 0.8% of price)
        # This avoids choppy low-volatility periods and focuses on meaningful moves
        if atr_14_1d_aligned[i] <= 0.008 * close[i]:
            signals[i] = 0.0
            continue
            
        # Long: Break above 20-period high with volume confirmation
        # Short: Break below 20-period low with volume confirmation
        if close[i] > highest_high_20[i]:
            # Volume confirmation: current volume > 1.3x 20-period average
            vol_ma_20 = pd.Series(volume[max(0, i-19):i+1]).mean()
            if volume[i] > 1.3 * vol_ma_20:
                signals[i] = 0.25  # 25% position
        elif close[i] < lowest_low_20[i]:
            # Volume confirmation: current volume > 1.3x 20-period average
            vol_ma_20 = pd.Series(volume[max(0, i-19):i+1]).mean()
            if volume[i] > 1.3 * vol_ma_20:
                signals[i] = -0.25  # 25% position
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian20_Volume_ATR_Regime_Filter_v1"
timeframe = "12h"
leverage = 1.0