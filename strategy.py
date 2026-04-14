#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using daily price channels and volume imbalance
# - Long when price breaks above daily Donchian upper band with volume > 1.5x 20-period average
# - Short when price breaks below daily Donchian lower band with volume > 1.5x 20-period average
# - Uses 4h ATR(14) for volatility filtering (ATR > 20-period ATR average)
# - Target: 80-160 total trades over 4 years (20-40/year) to balance opportunity and cost
# - Position size 0.25 for balanced risk exposure

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Load daily data once before loop
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 20:
        return np.zeros(n)
    
    # Calculate daily Donchian channels
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    
    # 20-period Donchian high/low
    donch_high = pd.Series(high_d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_d).rolling(window=20, min_periods=20).min().values
    
    # Daily volume average
    vol_d = df_d['volume'].values
    vol_ma_d = pd.Series(vol_d).rolling(window=20, min_periods=20).mean().values
    
    # Align daily indicators to 4h timeframe
    donch_high_4h = align_htf_to_ltf(prices, df_d, donch_high)
    donch_low_4h = align_htf_to_ltf(prices, df_d, donch_low)
    vol_ma_d_4h = align_htf_to_ltf(prices, df_d, vol_ma_d)
    
    # 4h ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_14 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(60, n):
        # Skip if any critical data is NaN
        if np.isnan(donch_high_4h[i]) or np.isnan(donch_low_4h[i]) or np.isnan(vol_ma_d_4h[i]) or np.isnan(atr_ma_14[i]):
            continue
            
        # Volatility filter: only trade when current ATR > average ATR
        if atr_14[i] <= atr_ma_14[i]:
            continue
            
        # Volume filter: current volume > 1.5x daily average volume
        if volume[i] <= vol_ma_d_4h[i] * 1.5:
            continue
            
        if position == 0:
            # Long: Price breaks above daily Donchian high with volume and volatility
            if high[i] > donch_high_4h[i]:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below daily Donchian low with volume and volatility
            elif low[i] < donch_low_4h[i]:
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Price returns to daily Donchian low or opposite band
            if low[i] <= donch_low_4h[i] or high[i] >= donch_high_4h[i]:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Price returns to daily Donchian high or opposite band
            if high[i] >= donch_high_4h[i] or low[i] <= donch_low_4h[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_1d_Donchian_Volume_Volatility"
timeframe = "4h"
leverage = 1.0