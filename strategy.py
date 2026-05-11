#!/usr/bin/env python3
name = "6h_ADX_Trend_Filter_EMA_Crossover"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA crossover (12 and 26) - base signal
    ema12 = pd.Series(close).ewm(span=12, min_periods=12, adjust=False).mean().values
    ema26 = pd.Series(close).ewm(span=26, min_periods=26, adjust=False).mean().values
    ema_cross = ema12 - ema26  # >0 = bullish momentum
    
    # ADX for trend strength (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM
    tr_smooth = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # DI values
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.nan_to_num(dx, nan=0.0)
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Daily trend filter from 1D EMA34
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter - 20-period average
    vol_ma = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema12[i]) or np.isnan(ema26[i]) or 
            np.isnan(adx[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        ema_bullish = ema12[i] > ema26[i]
        ema_bearish = ema12[i] < ema26[i]
        strong_trend = adx[i] > 25
        price_above_daily_ema = close[i] > ema34_1d_aligned[i]
        price_below_daily_ema = close[i] < ema34_1d_aligned[i]
        volume_ok = vol_ratio[i] > 1.2
        
        if position == 0:
            # Long: EMA bullish + strong trend + price above daily EMA + volume
            if ema_bullish and strong_trend and price_above_daily_ema and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: EMA bearish + strong trend + price below daily EMA + volume
            elif ema_bearish and strong_trend and price_below_daily_ema and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit: EMA turns bearish OR trend weakens OR price crosses below daily EMA
                if (not ema_bullish) or (adx[i] < 20) or (close[i] <= ema34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit: EMA turns bullish OR trend weakens OR price crosses above daily EMA
                if (not ema_bearish) or (adx[i] < 20) or (close[i] >= ema34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals