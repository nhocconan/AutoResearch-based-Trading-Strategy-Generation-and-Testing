#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_WilliamsVixFix_BullPower_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Williams Vix Fix (WVF) - Mean Reversion Signal ===
    # Measures market fear/greed, low = fear (potential bottom), high = greed (potential top)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate highest close over lookback period (22 days typical)
    lookback = 22
    highest_close = pd.Series(close_1d).rolling(window=lookback, min_periods=lookback).max().values
    
    # WVF = ((highest_close - low) / (highest_close - close)) * 100
    # Avoid division by zero
    denominator = highest_close - close_1d
    wvf = np.where(denominator != 0, ((highest_close - low_1d) / denominator) * 100, 100)
    
    # WVF < 50 indicates high fear (oversold), WVF > 150 indicates extreme greed (overbought)
    wf_oversold = wvf < 50
    wf_overbought = wvf > 150
    
    # Align WVF signals to 6h timeframe
    wf_oversold_aligned = align_htf_to_ltf(prices, df_1d, wf_oversold.astype(float))
    wf_overbought_aligned = align_htf_to_ltf(prices, df_1d, wf_overbought.astype(float))
    
    # === 1d Elder Ray Bull Power - Trend Strength ===
    # Bull Power = High - EMA(close)
    ema_length = 13
    ema_close = pd.Series(close_1d).ewm(span=ema_length, adjust=False, min_periods=ema_length).mean().values
    bull_power = high_1d - ema_close
    
    # Align Bull Power to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    
    # === 6h Volume Confirmation ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        vol_ratio_val = vol_ratio[i]
        wf_oversold_val = wf_oversold_aligned[i]
        wf_overbought_val = wf_overbought_aligned[i]
        bull_power_val = bull_power_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(wf_oversold_val) or np.isnan(wf_overbought_val) or 
            np.isnan(bull_power_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: WVF oversold (fear) + Bull Power > 0 (bullish momentum) + volume confirmation
            if wf_oversold_val and bull_power_val > 0 and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: WVF overbought (greed) + Bull Power < 0 (bearish momentum) + volume confirmation
            elif wf_overbought_val and bull_power_val < 0 and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        
        elif position == 1:
            # Long exit: WVF overbought (greed) or Bull Power turns negative
            if wf_overbought_val or bull_power_val <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: WVF oversold (fear) or Bull Power turns positive
            if wf_oversold_val or bull_power_val >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals