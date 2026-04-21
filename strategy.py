#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Fade_1dTrend_VolumeConfirm_v1
Hypothesis: On 6h timeframe, fade Camarilla R3/S3 levels (mean reversion) when 1d trend is strong (ADX>25) and volume confirms (>1.5x 20-period MA). 
In strong trends, price often pulls back to R3/S3 before continuing. Entry: price closes back inside R3/S3 after touching/exceeding it. 
Exit: price reaches opposite Camarilla level (R1/S1) or ATR stop (1.5x). Designed for low trade frequency (~15-25/year) to minimize fee drag while capturing mean reversion within strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla, trend, volume)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d OHLC for Camarilla pivot calculation ===
    df_1d_open = df_1d['open'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    range_1d = df_1d_high - df_1d_low
    r3_1d = df_1d_close + 0.55 * range_1d
    s3_1d = df_1d_close - 0.55 * range_1d
    r1_1d = df_1d_close + 0.275 * range_1d
    s1_1d = df_1d_close - 0.275 * range_1d
    
    # Align 1d Camarilla levels to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 1d ADX for trend strength filter ===
    # Calculate +DI, -DI, DX
    high = df_1d_high
    low = df_1d_low
    close = df_1d_close
    
    up_move = np.diff(high, prepend=high[0])
    down_move = np.diff(low, prepend=low[0]) * -1  # positive numbers
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / (atr_1d + 1e-10)
    minus_di_1d = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / (atr_1d + 1e-10)
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
    adx_1d = pd.Series(dx_1d).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 1d Volume average for confirmation ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # === 6h ATR for stoploss ===
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    tr1_6h = high_6h - low_6h
    tr2_6h = np.abs(high_6h - np.roll(close_6h, 1))
    tr3_6h = np.abs(low_6h - np.roll(close_6h, 1))
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) 
            or np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i])
            or np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) 
            or np.isnan(atr_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close_6h[i]
        volume_now = prices['volume'].values[i]
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        adx = adx_1d_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        atr = atr_6h[i]
        
        # Volume confirmation: current volume > 1.5x 1d average volume (scaled to 6h)
        # Approximate 6h volume as 1d/4 (since 4x 6h in 1d)
        vol_threshold = vol_ma * 0.375  # 1.5x * (1/4)
        volume_confirm = volume_now > vol_threshold
        
        # Strong trend filter: ADX > 25
        strong_trend = adx > 25
        
        if position == 0:
            # Fade R3/S3: price must have touched/exceeded level and now closed back inside
            # Long: price was >= r3 and now < r3
            # Short: price was <= s3 and now > s3
            # We approximate by checking if price crossed the level on this bar
            long_condition = (prices['high'].values[i-1] >= r3 and price < r3) or \
                            (close_6h[i-1] >= r3 and price < r3)
            short_condition = (prices['low'].values[i-1] <= s3 and price > s3) or \
                             (close_6h[i-1] <= s3 and price > s3)
            
            long_condition = long_condition and strong_trend and volume_confirm
            short_condition = short_condition and strong_trend and volume_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Exit conditions
            if position == 1:
                # Stoploss: 1.5x ATR
                if price < entry_price - 1.5 * atr:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Target: reach R1 (opposite Camarilla level)
                elif price >= r1:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Reversal: price touches S3 again (failed fade)
                elif price <= s3:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Stoploss: 1.5x ATR
                if price > entry_price + 1.5 * atr:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Target: reach S1 (opposite Camarilla level)
                elif price <= s1:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Reversal: price touches R3 again (failed fade)
                elif price >= r3:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3_S3_Fade_1dTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0