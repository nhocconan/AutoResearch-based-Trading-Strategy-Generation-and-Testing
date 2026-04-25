#!/usr/bin/env python3
"""
6h_AdaptiveDonchian_Volume_Regime
Hypothesis: On 6h timeframe, adaptive Donchian channels (ATR-scaled) combined with volume confirmation and regime filter (ADX < 25 for mean reversion, ADX > 25 for trend following) captures breakouts in trending markets and reversals in ranging markets. Uses 1d EMA50 for trend filter to avoid counter-trend trades. Designed for low trade frequency (~20-40/year) with discrete position sizing to minimize fee drag. Works in both bull and bear markets by adapting to regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for EMA50 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h ATR(20) for adaptive Donchian scaling and volatility
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # 6h adaptive Donchian channels: ATR-scaled lookback
    # Base lookback 20 periods, scaled by ATR ratio (current/mean ATR)
    atr_mean = pd.Series(atr_20).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_20 / atr_mean
    lookback = np.clip(20 * atr_ratio, 10, 50).astype(int)  # between 10 and 50
    
    # Calculate adaptive Donchian high/low using rolling window with dynamic lookback
    # We'll compute highest high and lowest low over the lookback period
    highest = np.full(n, np.nan)
    lowest = np.full(n, np.nan)
    
    for i in range(n):
        lb = int(lookback[i]) if not np.isnan(lookback[i]) else 20
        start_idx = max(0, i - lb + 1)
        if i >= lb - 1:
            highest[i] = np.max(high[start_idx:i+1])
            lowest[i] = np.min(low[start_idx:i+1])
    
    # 6h ADX(14) for regime filter
    # +DM and -DM
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # True Range (already calculated as 'tr')
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # +DI and -DI
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / tr_14
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # 6h volume spike: current volume > 1.8 * 30-period volume MA
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume > (1.8 * vol_ma_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need all indicators warmed up
    start_idx = max(50, 30, 20, 14)  # lookback MA, volume MA, ATR, ADX
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest[i]) or np.isnan(lowest[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_20[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Regime filter: ADX < 25 = ranging (mean reversion), ADX > 25 = trending (breakout)
            is_ranging = adx[i] < 25
            is_trending = adx[i] > 25
            
            # Long conditions
            long_breakout = (curr_close > highest[i]) and vol_spike[i] and is_trending and (curr_close > ema_50_1d_aligned[i])
            long_reversion = (curr_close < lowest[i]) and vol_spike[i] and is_ranging and (curr_close > ema_50_1d_aligned[i])
            
            # Short conditions
            short_breakout = (curr_close < lowest[i]) and vol_spike[i] and is_trending and (curr_close < ema_50_1d_aligned[i])
            short_reversion = (curr_close > highest[i]) and vol_spike[i] and is_ranging and (curr_close < ema_50_1d_aligned[i])
            
            if long_breakout or long_reversion:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout or short_reversion:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: opposite signal or volume spike reversal
            if (curr_close < lowest[i] and vol_spike[i]) or (curr_close < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: opposite signal or volume spike reversal
            if (curr_close > highest[i] and vol_spike[i]) or (curr_close > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_AdaptiveDonchian_Volume_Regime"
timeframe = "6h"
leverage = 1.0