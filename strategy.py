#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour True Strength Index (TSI) with daily volume confirmation and 1-day ADX trend filter
# TSI captures momentum with reduced whipsaw vs RSI/MACD. Long when TSI > 25 and rising, short when TSI < -25 and falling.
# Uses 1-day ADX > 25 to filter for trending conditions only, avoiding range-bound whipsaw.
# Volume confirmation (1-day volume > 1.5x 20-day average) ensures institutional participation.
# Designed for low trade frequency (~20-30/year) to minimize fee drag while capturing strong trends.
# Works in both bull (TSI > 25) and bear (TSI < -25) markets via symmetric long/short logic.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # === 1-day TSI (True Strength Index) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Price change and absolute price change
    pc = close_1d - np.concatenate([[close_1d[0]], close_1d[:-1]])
    apc = np.abs(pc)
    
    # Double smoothed PC and APC
    pc_sm1 = pd.Series(pc).ewm(span=25, adjust=False).mean()
    pc_sm2 = pc_sm1.ewm(span=13, adjust=False).mean()
    apc_sm1 = pd.Series(apc).ewm(span=25, adjust=False).mean()
    apc_sm2 = apc_sm1.ewm(span=13, adjust=False).mean()
    
    # TSI = 100 * (double smoothed PC / double smoothed APC)
    tsi_raw = 100 * (pc_sm2.values / apc_sm2.values)
    tsi_raw = np.where(apc_sm2.values == 0, 0, tsi_raw)  # avoid division by zero
    
    tsi_1d_aligned = align_htf_to_ltf(prices, df_1d, tsi_raw)
    
    # === 1-day ADX (14-period) for trend strength ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])
    down_move = np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM
    tr_sum = pd.Series(tr).ewm(span=14, adjust=False).mean()
    plus_dm_sum = pd.Series(plus_dm).ewm(span=14, adjust=False).mean()
    minus_dm_sum = pd.Series(minus_dm).ewm(span=14, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_sum / tr_sum
    minus_di = 100 * minus_dm_sum / tr_sum
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 1-day Volume Confirmation ===
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(tsi_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Trend and volume filters
        is_trending = adx_1d_aligned[i] > 25
        vol_confirmed = volume_1d[i] > vol_ma_20_aligned[i] * 1.5  # use current day's volume
        
        # Entry logic: only enter when flat
        if position == 0:
            if is_trending and vol_confirmed:
                # Long: TSI > 25 and rising (current > previous)
                if tsi_1d_aligned[i] > 25 and tsi_1d_aligned[i] > tsi_1d_aligned[i-1]:
                    signals[i] = 0.25
                    position = 1
                    continue
                # Short: TSI < -25 and falling (current < previous)
                elif tsi_1d_aligned[i] < -25 and tsi_1d_aligned[i] < tsi_1d_aligned[i-1]:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Exit logic
        elif position == 1:
            # Exit long when TSI drops below 0 (momentum fade)
            if tsi_1d_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when TSI rises above 0 (momentum fade)
            if tsi_1d_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_TSI_ADX_Volume_Trend_Filter"
timeframe = "4h"
leverage = 1.0