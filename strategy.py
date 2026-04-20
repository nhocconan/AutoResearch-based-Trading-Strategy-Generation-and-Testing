#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ADX_Trend_With_Pullback_Entry"
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
    
    # === 1d: Calculate ADX and EMA200 for trend filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values
    period = 14
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
    
    # EMA200 for trend direction
    ema200 = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    
    # Align to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200)
    
    # === 6h: Price and volume ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6-period EMA for pullback entry
    ema6 = pd.Series(close).ewm(span=6, adjust=False).mean().values
    
    # Volume filter: current vs 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        adx_val = adx_aligned[i]
        ema200_val = ema200_aligned[i]
        close_val = close[i]
        ema6_val = ema6[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(adx_val) or np.isnan(ema200_val) or np.isnan(ema6_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: ADX > 25 and price relative to EMA200
        is_uptrend = (adx_val > 25) and (close_val > ema200_val)
        is_downtrend = (adx_val > 25) and (close_val < ema200_val)
        
        if position == 0:
            # Long: Pullback in uptrend to EMA6 with volume
            if is_uptrend and (close_val <= ema6_val) and (vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Pullback in downtrend to EMA6 with volume
            elif is_downtrend and (close_val >= ema6_val) and (vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Trend weakness or mean reversion
            if (adx_val < 20) or (close_val < ema6_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Trend weakness or mean reversion
            if (adx_val < 20) or (close_val > ema6_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals