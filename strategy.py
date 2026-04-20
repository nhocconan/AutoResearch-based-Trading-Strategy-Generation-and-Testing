#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h ADX + Parabolic SAR trend following with volume filter
# - Use ADX(14) > 25 to identify trending markets
# - Use Parabolic SAR with acceleration factor 0.02, max 0.2 for trend direction
# - Long when SAR < close and ADX > 25 with volume > 1.3x 20-period average
# - Short when SAR > close and ADX > 25 with volume > 1.3x 20-period average
# - Exit when SAR flips or ADX drops below 20
# - Uses 1d ADX for stronger trend confirmation and 4h for execution
# - Target: 20-35 trades per year per symbol (80-140 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX on 1d data
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di14 = np.where(tr14 != 0, 100 * plus_dm14 / tr14, 0)
    minus_di14 = np.where(tr14 != 0, 100 * minus_dm14 / tr14, 0)
    
    # DX and ADX
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Parabolic SAR on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Initialize SAR
    sar = np.zeros(n)
    trend = np.ones(n)  # 1 for uptrend, -1 for downtrend
    af = 0.02  # acceleration factor
    max_af = 0.2
    ep = high[0]  # extreme point
    
    sar[0] = low[0]
    
    for i in range(1, n):
        if trend[i-1] == 1:  # uptrend
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            if sar[i] > low[i]:
                trend[i] = -1
                sar[i] = ep
                af = 0.02
                ep = low[i]
            else:
                trend[i] = 1
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + 0.02, max_af)
        else:  # downtrend
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            if sar[i] < high[i]:
                trend[i] = 1
                sar[i] = ep
                af = 0.02
                ep = high[i]
            else:
                trend[i] = -1
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + 0.02, max_af)
    
    # Volume confirmation: 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if np.isnan(adx_1d_aligned[i]) or np.isnan(sar[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: SAR below close, ADX > 25, volume surge
            if sar[i] < price and adx_1d_aligned[i] > 25 and vol > 1.3 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: SAR above close, ADX > 25, volume surge
            elif sar[i] > price and adx_1d_aligned[i] > 25 and vol > 1.3 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: SAR flips above close OR ADX drops below 20
            if sar[i] > price or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: SAR flips below close OR ADX drops below 20
            if sar[i] < price or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_ADX_SAR_VolumeTrend"
timeframe = "4h"
leverage = 1.0