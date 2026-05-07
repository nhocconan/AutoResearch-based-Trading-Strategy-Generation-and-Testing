#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h ATR-based breakout with 1d volume surge and ADX trend filter.
# Long when price breaks above ATR(14) upper band AND volume surge AND ADX > 25 (trending).
# Short when price breaks below ATR(14) lower band AND volume surge AND ADX > 25.
# Uses daily volume surge for momentum confirmation and ADX to avoid ranging markets.
# Designed for moderate trade frequency (target: 20-40/year) to balance opportunity and cost.
# Works in both bull and bear markets by capturing volatility breakouts with trend confirmation.
name = "12h_ATR_Breakout_Volume_ADX"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for volume surge and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d volume surge: current volume > 2.0 * 20-period SMA
    vol_sma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_surge_1d = np.where(vol_sma_20 > 0, df_1d['volume'].values / vol_sma_20, 1.0) > 2.0
    vol_surge_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_surge_1d)
    
    # 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr_list = []
    for i in range(len(close_1d)):
        if i == 0:
            tr = high_1d[0] - low_1d[0]
        else:
            tr = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        tr_list.append(tr)
    tr = np.array(tr_list)
    
    # Directional Movement
    dm_plus = np.zeros(len(close_1d))
    dm_minus = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        dm_plus[i] = up_move if up_move > down_move and up_move > 0 else 0
        dm_minus[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus = np.where(tr_14 > 0, 100 * dm_plus_14 / tr_14, 0)
    di_minus = np.where(tr_14 > 0, 100 * dm_minus_14 / tr_14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Load 12h data for ATR bands
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Previous 12h OHLC for ATR calculation
    prev_close_12h = df_12h['close'].shift(1).values
    prev_high_12h = df_12h['high'].shift(1).values
    prev_low_12h = df_12h['low'].shift(1).values
    
    # True Range for 12h
    tr_12h_list = []
    for i in range(len(prev_close_12h)):
        if i == 0:
            tr = prev_high_12h[0] - prev_low_12h[0]
        else:
            tr = max(prev_high_12h[i] - prev_low_12h[i], abs(prev_high_12h[i] - prev_close_12h[i-1]), abs(prev_low_12h[i] - prev_close_12h[i-1]))
        tr_12h_list.append(tr)
    tr_12h = np.array(tr_12h_list)
    
    # ATR(10) for 12h
    atr_10_12h = pd.Series(tr_12h).rolling(window=10, min_periods=10).mean().values
    
    # ATR Bands: midpoint = average of previous high/low
    midpoint_12h = (prev_high_12h + prev_low_12h) / 2
    atr_upper = midpoint_12h + 1.5 * atr_10_12h
    atr_lower = midpoint_12h - 1.5 * atr_10_12h
    
    atr_upper_aligned = align_htf_to_ltf(prices, df_12h, atr_upper)
    atr_lower_aligned = align_htf_to_ltf(prices, df_12h, atr_lower)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(atr_upper_aligned[i]) or np.isnan(atr_lower_aligned[i]) or 
            np.isnan(vol_surge_1d_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long condition: break above ATR upper band, volume surge, trending (ADX > 25)
            long_condition = (close[i] > atr_upper_aligned[i]) and vol_surge_1d_aligned[i] and (adx_1d_aligned[i] > 25)
            # Short condition: break below ATR lower band, volume surge, trending (ADX > 25)
            short_condition = (close[i] < atr_lower_aligned[i]) and vol_surge_1d_aligned[i] and (adx_1d_aligned[i] > 25)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below ATR lower band or ADX drops to ranging (ADX < 20)
            if (close[i] < atr_lower_aligned[i]) or (adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above ATR upper band or ADX drops to ranging (ADX < 20)
            if (close[i] > atr_upper_aligned[i]) or (adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals