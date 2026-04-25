#!/usr/bin/env python3
"""
6h_ADX_Alligator_Trend_Regime_Volume
Hypothesis: Combines ADX trend strength with Williams Alligator (smoothed medians) to identify strong trends, filters with 1w EMA50 for higher timeframe direction, and requires volume confirmation. The Alligator's Jaw/Teeth/Lips provide dynamic support/resistance while ADX > 25 ensures we only trade in trending markets. Works in both bull/bear via 1w trend filter and discrete sizing (0.25) to manage drawdown. Targets 12-30 trades/year by requiring confluence of multiple trend signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Williams Alligator (Jaw/Teeth/Lips)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    median_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    
    # Williams Alligator: Jaw (13-period, 8 bars ahead), Teeth (8-period, 5 bars ahead), Lips (5-period, 3 bars ahead)
    # Using SMMA (Smoothed Moving Average) which is EMA with alpha=1/period
    jaw = pd.Series(median_1d).ewm(alpha=1/13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(median_1d).ewm(alpha=1/8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(median_1d).ewm(alpha=1/5, adjust=False, min_periods=5).mean().values
    
    # Shift forward: Jaw +8, Teeth +5, Lips +3 (to avoid look-ahead)
    jaw = np.concatenate([np.full(8, np.nan), jaw[:-8]]) if len(jaw) > 8 else np.full_like(jaw, np.nan)
    teeth = np.concatenate([np.full(5, np.nan), teeth[:-5]]) if len(teeth) > 5 else np.full_like(teeth, np.nan)
    lips = np.concatenate([np.full(3, np.nan), lips[:-3]]) if len(lips) > 3 else np.full_like(lips, np.nan)
    
    # Align Alligator lines to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate ADX(14) on 6h for trend strength
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Volume average (20-period) for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(50, 20, 14, 13, 8, 5)  # EMA50_1w, vol MA, ADX, Alligator
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(adx[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(plus_di[i]) or 
            np.isnan(minus_di[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_1w_val = ema_50_1w_aligned[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        adx_val = adx[i]
        plus_di_val = plus_di[i]
        minus_di_val = minus_di[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        
        # Trend filter: 1w EMA50 direction
        uptrend_1w = close_val > ema_1w_val
        downtrend_1w = close_val < ema_1w_val
        
        # Alligator alignment: 
        # Bullish: Lips > Teeth > Jaw (green alignment)
        # Bearish: Lips < Teeth < Jaw (red alignment)
        alligator_bullish = lips_val > teeth_val > jaw_val
        alligator_bearish = lips_val < teeth_val < jaw_val
        
        # ADX trend strength: > 25 indicates strong trend
        strong_trend = adx_val > 25
        
        # Directional movement confirmation
        bullish_dmi = plus_di_val > minus_di_val
        bearish_dmi = minus_di_val > plus_di_val
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = vol_val > 1.5 * vol_ma_val
        
        if position == 0:
            # Look for entry signals: Alligator alignment + ADX + DMI + 1w trend + volume
            long_signal = (alligator_bullish and 
                          strong_trend and 
                          bullish_dmi and 
                          uptrend_1w and 
                          volume_confirmed)
            
            short_signal = (alligator_bearish and 
                           strong_trend and 
                           bearish_dmi and 
                           downtrend_1w and 
                           volume_confirmed)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Alligator reverses (Lips < Teeth)
            if lips_val < teeth_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. ADX weakens (< 20) - trend losing strength
            elif adx_val < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 3. 1w trend reverses (price crosses below EMA50)
            elif close_val < ema_1w_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Alligator reverses (Lips > Teeth)
            if lips_val > teeth_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. ADX weakens (< 20) - trend losing strength
            elif adx_val < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 3. 1w trend reverses (price crosses above EMA50)
            elif close_val > ema_1w_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "6h_ADX_Alligator_Trend_Regime_Volume"
timeframe = "6h"
leverage = 1.0