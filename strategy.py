#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 level breakout with 1d volume confirmation and 1d volatility regime filter.
# Long when price breaks above R3 (resistance) with volume > 1.5x daily average and daily volatility (ATR14) < ATR(50)
# Short when price breaks below S3 (support) with volume > 1.5x daily average and daily volatility (ATR14) < ATR(50)
# Exit when price crosses back through the daily pivot point
# Uses Camarilla levels from daily timeframe for institutional reference points, volume for confirmation, volatility regime to avoid chop.
# Target: 20-30 trades/year per symbol.

name = "4h_Camarilla_R3S3_Volume_VolatilityRegime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels, volume average, and volatility regime
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate True Range components for daily ATR
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Calculate daily average volume (20-period)
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla formulas: 
    # H = high, L = low, C = close of previous day
    # R4 = C + (H-L)*1.5/2, R3 = C + (H-L)*1.25/2, R2 = C + (H-L)*1.1/2, R1 = C + (H-L)*0.5/2
    # S1 = C - (H-L)*0.5/2, S2 = C - (H-L)*1.1/2, S3 = C - (H-L)*1.25/2, S4 = C - (H-L)*1.5/2
    # Pivot = (H+L+C)/3
    H = df_1d['high'].values
    L = df_1d['low'].values
    C = df_1d['close'].values
    
    # Calculate levels using previous day's values (shift by 1 to avoid look-ahead)
    H_prev = np.roll(H, 1)
    L_prev = np.roll(L, 1)
    C_prev = np.roll(C, 1)
    H_prev[0] = np.nan  # First value has no previous day
    L_prev[0] = np.nan
    C_prev[0] = np.nan
    
    # Calculate Camarilla levels
    R3 = C_prev + (H_prev - L_prev) * 1.25 / 2
    S3 = C_prev - (H_prev - L_prev) * 1.25 / 2
    pivot = (H_prev + L_prev + C_prev) / 3
    
    # Align all 1d indicators to 4h timeframe
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    atr50_aligned = align_htf_to_ltf(prices, df_1d, atr50)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr14_aligned[i]) or np.isnan(atr50_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr14_val = atr14_aligned[i]
        atr50_val = atr50_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        vol = volume[i]
        r3 = R3_aligned[i]
        s3 = S3_aligned[i]
        pivot_val = pivot_aligned[i]
        
        # Volatility regime filter: only trade in low volatility (ATR14 < ATR50)
        vol_regime = atr14_val < atr50_val
        
        if position == 0:
            # Long entry: break above R3 + volume spike + low vol regime
            if price > r3 and vol > 1.5 * vol_ma and vol_regime:
                signals[i] = 0.25
                position = 1
            # Short entry: break below S3 + volume spike + low vol regime
            elif price < s3 and vol > 1.5 * vol_ma and vol_regime:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below pivot
            if price < pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above pivot
            if price > pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals