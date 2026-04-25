#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_Confluence_v3
Hypothesis: Camarilla R1/S1 breakout on 4h with 1d EMA34 trend filter, volume spike (>1.8x avg), and ADX>25 trend confirmation.
Only trade breakouts aligned with 1d EMA34 direction during strong trending markets (ADX>25) with volume expansion.
Designed to reduce overtrading while capturing high-probability momentum moves in both bull and bear regimes.
Uses discrete sizing (0.25) and ATR-based stoploss (signal→0 when price moves against position by 2.0*ATR).
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
    
    # Get 4h data for Camarilla levels - primary timeframe
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for R1 and S1
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    camarilla_range = high_4h - low_4h
    r1_4h = close_4h + camarilla_range * 1.1 / 12
    s1_4h = close_4h - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # Get 1d data for EMA34 trend filter - HTF
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate ATR(14) for stoploss on 4h
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_4h = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # Calculate ADX(14) on 4h for trend strength filter
    # +DM and -DM
    up_move = high_4h[1:] - high_4h[:-1]
    down_move = low_4h[:-1] - low_4h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed +DM, -DM, and TR
    tr_sum = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / tr_sum
    minus_di = 100 * minus_dm_smooth / tr_sum
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(50, 34, 21, 20, 28)  # Camarilla needs 2, EMA needs 34, ATR needs 14, vol needs 20, ADX needs ~28
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or 
            np.isnan(atr_4h_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(adx_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_1d_aligned[i]
        atr_val = atr_4h_aligned[i]
        vol_ma_val = vol_ma[i]
        adx_val = adx_aligned[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current volume > 1.8x 20-period average (stricter to reduce trades)
        volume_spike = vol_val > 1.8 * vol_ma_val
        
        # Trend strength condition: ADX > 25 indicates strong trend
        strong_trend = adx_val > 25
        
        if position == 0:
            # Look for entry signals: Camarilla breakout with trend, volume, and ADX confirmation
            # Long: price breaks above R1, above 1d EMA34, with volume spike and strong trend
            long_signal = (high_val > r1_val) and (close_val > ema_val) and volume_spike and strong_trend
            # Short: price breaks below S1, below 1d EMA34, with volume spike and strong trend
            short_signal = (low_val < s1_val) and (close_val < ema_val) and volume_spike and strong_trend
            
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
            # 1. Stoploss: price moves against position by 2.0*ATR (wider stop to reduce whipsaw)
            if close_val < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: price closes below 1d EMA34
            elif close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Stoploss: price moves against position by 2.0*ATR
            if close_val > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: price closes above 1d EMA34
            elif close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Confluence_v3"
timeframe = "4h"
leverage = 1.0