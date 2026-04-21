#!/usr/bin/env python3
"""
6h_EMA_Cross_Volume_Regime_v1
Hypothesis: On 6h timeframe, EMA(9)/EMA(21) cross with volume confirmation and ADX regime filter captures medium-term momentum. Long on bullish cross with rising volume in trending/accumulation regime (ADX>20); short on bearish cross with rising volume in trending/distribution regime. Designed for low trade frequency (12-37/year) to minimize fee drag and work in both bull (continuation) and bear (mean reversion via short) regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for volume regime context)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # === 1-day average volume for regime filter ===
    volume_1d = df_1d['volume'].values
    avg_volume_1d = pd.Series(volume_1d).rolling(window=21, min_periods=21).mean().values
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # === EMA(9) and EMA(21) on 6h close ===
    close = prices['close'].values
    ema_9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # === ADX(14) for regime filter ===
    high = prices['high'].values
    low = prices['low'].values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    tr_smooth = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_9[i]) or np.isnan(ema_21[i]) or 
            np.isnan(adx[i]) or np.isnan(avg_volume_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_open = prices['open'].iloc[i]
        volume = prices['volume'].iloc[i]
        ema_9_val = ema_9[i]
        ema_21_val = ema_21[i]
        adx_val = adx[i]
        avg_vol_1d = avg_volume_1d_aligned[i]
        
        # Volume confirmation: current 6h volume > 1.5x daily average volume (scaled)
        # Daily avg volume approximated for 6h: divide by 4 (since 4x 6h in 1d)
        vol_threshold = avg_vol_1d * 0.375  # 1.5x / 4 = 0.375
        volume_confirmed = volume > vol_threshold
        
        if position == 0:
            # Bullish EMA cross: EMA9 crosses above EMA21
            bullish_cross = ema_9_val > ema_21_val and ema_9[i-1] <= ema_21[i-1]
            # Bearish EMA cross: EMA9 crosses below EMA21
            bearish_cross = ema_9_val < ema_21_val and ema_9[i-1] >= ema_21[i-1]
            
            # Long: bullish cross + volume confirmed + ADX > 20 (trending or accumulation)
            if bullish_cross and volume_confirmed and adx_val > 20:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Short: bearish cross + volume confirmed + ADX > 20 (trending or distribution)
            elif bearish_cross and volume_confirmed and adx_val > 20:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Stoploss: 2.5 * ATR from entry
            if position == 1:
                if price_close < entry_price - 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price_close > entry_price + 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_EMA_Cross_Volume_Regime_v1"
timeframe = "6h"
leverage = 1.0