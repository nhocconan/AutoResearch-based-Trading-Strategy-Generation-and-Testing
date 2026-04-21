#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily EMA21 for trend filter
    close_1d = df_1d['close'].values
    ema_21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # Daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d_arr, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d_arr, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Daily Bollinger Bands (20,2) for volatility regime
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    
    # Volume ratio: current volume / 20-day average volume
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = df_1d['volume'].values / vol_ma_20
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(ema_21_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(bb_width_aligned[i]) or np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_21_1d_aligned[i]
        atr = atr_1d_aligned[i]
        bb_width_val = bb_width_aligned[i]
        vol_ratio_val = vol_ratio_aligned[i]
        
        # Entry conditions: price near daily EMA21 with volume spike in low volatility
        near_ema = abs(price_close - ema_trend) < (0.5 * atr)
        vol_spike = vol_ratio_val > 1.5
        low_volatility = bb_width_val < 0.04  # Bollinger Band width < 4%
        
        if position == 0:
            # Enter long: price above EMA21, volume spike, low volatility
            if price_close > ema_trend and vol_spike and low_volatility:
                signals[i] = 0.25
                position = 1
            # Enter short: price below EMA21, volume spike, low volatility
            elif price_close < ema_trend and vol_spike and low_volatility:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price moves away from EMA21 or volatility increases
            if position == 1 and (price_close < ema_trend - (1.0 * atr) or bb_width_val > 0.06):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > ema_trend + (1.0 * atr) or bb_width_val > 0.06):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4d_EMA21_VolumeSpike_LowVol"
timeframe = "4h"
leverage = 1.0