#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_RSI_Slope_Divergence_TrendFollow"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # === 1W: RSI(14) with slope detection for trend strength ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # RSI calculation
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # RSI slope (3-period linear regression slope)
    rsi_slope = np.full_like(rsi_1w, np.nan)
    for i in range(14, len(rsi_1w)):
        y = rsi_1w[i-2:i+1]
        x = np.arange(3)
        if not np.any(np.isnan(y)):
            slope = np.polyfit(x, y, 1)[0]
            rsi_slope[i] = slope
    
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    rsi_slope_aligned = align_htf_to_ltf(prices, df_1w, rsi_slope)
    
    # === 1D: ADX(14) for trend strength filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / np.where(atr == 0, 1e-10, atr)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / np.where(atr == 0, 1e-10, atr)
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1e-10, (plus_di + minus_di))
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 6H: Price action and volume confirmation ===
    close = prices['close'].values
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        rsi_val = rsi_1w_aligned[i]
        rsi_slope_val = rsi_slope_aligned[i]
        adx_val = adx_aligned[i]
        vol_ratio_val = vol_ratio[i]
        close_val = close[i]
        
        if (np.isnan(rsi_val) or np.isnan(rsi_slope_val) or 
            np.isnan(adx_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Entry conditions
        if position == 0:
            # Long: RSI > 50 and rising, ADX > 25, volume confirmation
            if (rsi_val > 50 and rsi_slope_val > 0.5 and 
                adx_val > 25 and vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: RSI < 50 and falling, ADX > 25, volume confirmation
            elif (rsi_val < 50 and rsi_slope_val < -0.5 and 
                  adx_val > 25 and vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI declines below 50 or slope turns negative
            if rsi_val < 50 or rsi_slope_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI rises above 50 or slope turns positive
            if rsi_val > 50 or rsi_slope_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals