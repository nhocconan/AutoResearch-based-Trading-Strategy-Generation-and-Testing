#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band breakout with volume confirmation and ATR-based volatility filter
# Long when price breaks above upper BB(20,2) + volume > 1.3x avg + ATR(14) < ATR(50) (low volatility breakout)
# Short when price breaks below lower BB(20,2) + volume > 1.3x avg + ATR(14) < ATR(50)
# Uses 1d HTF for regime filter: only trade when 1d ADX < 25 (ranging/low trend market)
# Designed for low trade frequency (15-25/year) to minimize fee drag while capturing volatility expansion moves

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # === 4h Indicators: Bollinger Bands (20,2) ===
    close_4h = df_4h['close'].values
    bb_middle = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_4h).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + (2 * bb_std)
    bb_lower = bb_middle - (2 * bb_std)
    bb_upper_aligned = align_htf_to_ltf(prices, df_4h, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_4h, bb_lower)
    
    # === 4h Indicators: ATR for volatility filter ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range calculation
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_4h[0] - low_4h[0]  # first TR
    
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_4h, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_4h, atr_50)
    
    # === 1d Indicators: ADX for regime filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for ADX
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = high_1d[0] - low_1d[0]
    
    # Directional Movement
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di_14 = 100 * plus_dm_14 / np.where(tr_14 == 0, 1e-10, tr_14)
    minus_di_14 = 100 * minus_dm_14 / np.where(tr_14 == 0, 1e-10, tr_14)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / np.where((plus_di_14 + minus_di_14) == 0, 1e-10, (plus_di_14 + minus_di_14))
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.3x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # Volatility filter: ATR(14) < ATR(50) (low volatility environment)
        vol_filter = atr_14_aligned[i] < atr_50_aligned[i]
        
        # Regime filter: 1d ADX < 25 (ranging/low trend market)
        regime_filter = adx_aligned[i] < 25
        
        # Skip if any required data is NaN
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or
            np.isnan(vol_sma_20[i]) or np.isnan(atr_14_aligned[i]) or
            np.isnan(atr_50_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above upper Bollinger Band
        # 2. Volume confirmation
        # 3. Low volatility environment (volatility contraction breakout)
        # 4. Ranging market regime (ADX < 25)
        if (close[i] > bb_upper_aligned[i]) and vol_confirm and vol_filter and regime_filter:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below lower Bollinger Band
        # 2. Volume confirmation
        # 3. Low volatility environment (volatility contraction breakout)
        # 4. Ranging market regime (ADX < 25)
        elif (close[i] < bb_lower_aligned[i]) and vol_confirm and vol_filter and regime_filter:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_BB20_Volume_Volatility_ADX_Filter_v1"
timeframe = "4h"
leverage = 1.0