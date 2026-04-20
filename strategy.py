#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-week period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    upper_band = np.full(len(high_1w), np.nan)
    lower_band = np.full(len(low_1w), np.nan)
    for i in range(19, len(high_1w)):
        upper_band[i] = np.max(high_1w[i-19:i+1])
        lower_band[i] = np.min(low_1w[i-19:i+1])
    
    # Align to daily timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1w, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1w, lower_band)
    
    # Calculate daily ATR for volatility filter
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = np.zeros_like(tr)
    for i in range(14, len(tr)):
        atr[i] = np.mean(tr[i-14:i+1])
    atr[:14] = np.nan
    
    # Calculate daily volume average
    volume = prices['volume'].values
    vol_ma = np.zeros_like(volume)
    for i in range(20, len(volume)):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_ma[:20] = np.nan
    
    # Generate signals
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if NaN
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        atr_val = atr[i]
        vol_ma_val = vol_ma[i]
        upper = upper_band_aligned[i]
        lower = lower_band_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper band + volume spike + volatility filter
            if price > upper and vol > 1.5 * vol_ma_val and atr_val > 0.5 * np.mean(atr[max(0, i-50):i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band + volume spike + volatility filter
            elif price < lower and vol > 1.5 * vol_ma_val and atr_val > 0.5 * np.mean(atr[max(0, i-50):i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price drops below lower band or ATR drops
            if price < lower or atr_val < 0.3 * np.mean(atr[max(0, i-50):i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above upper band or ATR drops
            if price > upper or atr_val < 0.3 * np.mean(atr[max(0, i-50):i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchian_Breakout_Volume_ATRFilter_v1"
timeframe = "1d"
leverage = 1.0