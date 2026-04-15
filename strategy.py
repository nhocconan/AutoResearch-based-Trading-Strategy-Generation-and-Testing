#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d ATR regime filter
# Long when price breaks above Donchian upper (20) + volume > 1.5x 20-period avg + ATR(14) < ATR(50) (low vol regime)
# Short when price breaks below Donchian lower (20) + volume > 1.5x 20-period avg + ATR(14) < ATR(50) (low vol regime)
# Uses 1d ATR for regime filter to avoid breakouts during high volatility (false breakouts)
# Designed for low trade frequency (15-40/year) to minimize fee drag while capturing genuine breakouts

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: ATR for Regime Filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first period
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # first period
    tr3[0] = np.abs(low_1d[0] - close_1d[0])  # first period
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50_1d = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_50_1d)
    
    # === 4h Indicators: Donchian Channels (20-period) ===
    donch_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume SMA (20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Regime filter: ATR(14) < ATR(50) (low volatility regime)
        vol_regime = (not np.isnan(atr_14_1d_aligned[i]) and 
                     not np.isnan(atr_50_1d_aligned[i]) and
                     atr_14_1d_aligned[i] < atr_50_1d_aligned[i])
        
        # Skip if any required data is NaN
        if (np.isnan(donch_high_20[i]) or np.isnan(donch_low_20[i]) or
            np.isnan(vol_sma_20[i]) or not vol_confirm or not vol_regime):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 4h Donchian upper (20)
        # 2. Volume confirmation
        # 3. Low volatility regime (ATR14 < ATR50)
        if (close[i] > donch_high_20[i]) and vol_confirm and vol_regime:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 4h Donchian lower (20)
        # 2. Volume confirmation
        # 3. Low volatility regime (ATR14 < ATR50)
        elif (close[i] < donch_low_20[i]) and vol_confirm and vol_regime:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_Volume_LowVolATR_Filter_v1"
timeframe = "4h"
leverage = 1.0