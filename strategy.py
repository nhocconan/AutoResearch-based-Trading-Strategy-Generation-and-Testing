# 6h_1d_Camarilla_R1S1_Breakout_Volume_ATRFilter_v1
# Hypothesis: On 6h timeframe, price breaking above R1 or below S1 of daily Camarilla pivot with volume spike and ATR filter captures institutional breakouts in both bull and bear markets.
# Volume spike filters false breakouts, ATR filter avoids choppy markets. Targets 15-25 trades/year.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Camarilla_R1S1_Breakout_Volume_ATRFilter_v1"
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
    
    # Get daily data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla pivot levels from previous day
    prev_close = np.roll(close_1d, 1)
    prev_close[0] = np.nan
    prev_high = np.roll(high_1d, 1)
    prev_high[0] = np.nan
    prev_low = np.roll(low_1d, 1)
    prev_low[0] = np.nan
    
    # Pivot = (H + L + C) / 3
    pivot = (prev_high + prev_low + prev_close) / 3.0
    # R1 = C + (H - L) * 1.1 / 12
    r1 = prev_close + (prev_high - prev_low) * 1.1 / 12.0
    # S1 = C - (H - L) * 1.1 / 12
    s1 = prev_close - (prev_high - prev_low) * 1.1 / 12.0
    
    # Align to 6h timeframe
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR filter: avoid choppy markets (ATR < 0.5 * 20-period ATR mean)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_20 = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Need 20 for vol MA, 14+20 for ATR
    
    for i in range(start_idx, n):
        if np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr_ma_20[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        atr_ma = atr_ma_20[i]
        
        # Volume spike: current volume > 1.5x average
        volume_spike = vol > 1.5 * vol_ma
        # ATR filter: only trade when volatility is elevated (ATR > 0.5 * 20-period ATR mean)
        atr_filter = atr_val > 0.5 * atr_ma
        
        if position == 0:
            # Long: Price breaks above R1 with volume spike and ATR filter
            if price > r1_6h[i] and volume_spike and atr_filter:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume spike and ATR filter
            elif price < s1_6h[i] and volume_spike and atr_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below S1 (reversal signal)
            if price < s1_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above R1 (reversal signal)
            if price > r1_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals