#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(40) breakout with 1d volume spike (>2x average) and ADX(14) > 25
# Breakouts capture momentum; volume surge confirms institutional interest; ADX filters chop.
# Works in bull/bear by trading breakouts in direction of price vs 50 EMA on 1d.
# Position size: 0.25 for balanced risk/return.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for volume, ADX, and EMA ===
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d ADX calculation (14-period)
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(low_1d)
    for i in range(1, len(high_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / (atr_1d + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / (atr_1d + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 1d volume and its 20-period average
    volume_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma20_1d)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 40-period Donchian channels on 4h
    donchian_high_4h = pd.Series(high).rolling(window=40, min_periods=40).max().values
    donchian_low_4h = pd.Series(low).rolling(window=40, min_periods=40).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if any required data is not available
        if np.isnan(adx_1d_aligned[i]) or np.isnan(volume_ma20_1d_aligned[i]) or \
           np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 2x 20-period average volume
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        volume_filter = vol_1d_current > (2.0 * volume_ma20_1d_aligned[i])
        
        # Trend filter: ADX > 25 indicates trending market
        trend_filter = adx_1d_aligned[i] > 25
        
        # Price trend filter: close vs 1d EMA50
        price_above_ema = close_1d[-1] > ema_50_1d_aligned[i] if len(close_1d) > 0 else False
        
        # Combined filter
        filter_ok = volume_filter and trend_filter
        
        if position == 0:
            # Long when price breaks above 40-period Donchian high AND price above EMA50
            if (close[i] > donchian_high_4h[i] and 
                price_above_ema and filter_ok):
                signals[i] = 0.25
                position = 1
            # Short when price breaks below 40-period Donchian low AND price below EMA50
            elif (close[i] < donchian_low_4h[i] and 
                  not price_above_ema and filter_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 40-period Donchian low or filter fails
            if (close[i] < donchian_low_4h[i] or 
                not filter_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above 40-period Donchian high or filter fails
            if (close[i] > donchian_high_4h[i] or 
                not filter_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian40_1dVolumeSpike_ADX_EMA50"
timeframe = "4h"
leverage = 1.0