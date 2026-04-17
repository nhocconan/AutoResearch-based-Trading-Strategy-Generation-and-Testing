#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_RangeBound_MeanReversion
Hypothesis: In ranging markets, price tends to revert from Camarilla R1/S1 levels on 12h timeframe.
Long near S1 when price is below EMA34 (bearish bias) with bullish reversal signals.
Short near R1 when price is above EMA34 (bullish bias) with bearish reversal signals.
Uses RSI(7) for mean reversion signals and volume confirmation. Designed for 12h to avoid overtrading.
Works in sideways markets (2025-2026) and captures reversals from extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for Camarilla pivot, EMA trend, and RSI ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels from previous 1d bar
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    
    # Handle first bar
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    prev_close_1d[0] = close_1d[0]
    
    cam_r1 = prev_close_1d + 0.25 * (prev_high_1d - prev_low_1d)
    cam_s1 = prev_close_1d - 0.25 * (prev_high_1d - prev_low_1d)
    
    # Align Camarilla levels to 12h timeframe
    cam_r1_aligned = align_htf_to_ltf(prices, df_1d, cam_r1)
    cam_s1_aligned = align_htf_to_ltf(prices, df_1d, cam_s1)
    
    # 1d EMA34 for trend bias filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d RSI(7) for mean reversion signals
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/7, min_periods=7, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/7, min_periods=7, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 1d volume average (20-period) for volume confirmation
    vol_avg20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg20_1d)
    
    signals = np.zeros(n)
    
    # Warmup: covers EMA34, RSI, and rollouts
    warmup = 40
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(cam_r1_aligned[i]) or 
            np.isnan(cam_s1_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_avg20_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 1d volume
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        
        # Volume filter: current volume > 1.3x 20-period average
        vol_filter = vol_1d_current > 1.3 * vol_avg20_1d_aligned[i]
        
        # Mean reversion conditions
        if position == 0:
            # Long near S1: price near support, bearish bias (price < EMA34), oversold RSI
            price_near_s1 = abs(close[i] - cam_s1_aligned[i]) < 0.005 * close[i]  # within 0.5%
            bearish_bias = close[i] < ema34_1d_aligned[i]
            oversold = rsi_1d_aligned[i] < 30
            
            if price_near_s1 and bearish_bias and oversold and vol_filter:
                signals[i] = 0.25
                position = 1
                continue
            
            # Short near R1: price near resistance, bullish bias (price > EMA34), overbought RSI
            price_near_r1 = abs(close[i] - cam_r1_aligned[i]) < 0.005 * close[i]  # within 0.5%
            bullish_bias = close[i] > ema34_1d_aligned[i]
            overbought = rsi_1d_aligned[i] > 70
            
            if price_near_r1 and bullish_bias and overbought and vol_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions: mean reversion complete or RSI returns to neutral
        elif position == 1:
            # Exit long when RSI returns to neutral or price moves to midpoint
            rsi_neutral = rsi_1d_aligned[i] > 50
            price_at_mid = abs(close[i] - (cam_r1_aligned[i] + cam_s1_aligned[i])/2) < 0.002 * close[i]
            
            if rsi_neutral or price_at_mid:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when RSI returns to neutral or price moves to midpoint
            rsi_neutral = rsi_1d_aligned[i] < 50
            price_at_mid = abs(close[i] - (cam_r1_aligned[i] + cam_s1_aligned[i])/2) < 0.002 * close[i]
            
            if rsi_neutral or price_at_mid:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_RangeBound_MeanReversion"
timeframe = "12h"
leverage = 1.0