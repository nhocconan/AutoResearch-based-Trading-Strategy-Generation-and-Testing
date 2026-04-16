#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels (R1/S1) with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above S1 (support) in ranging market AND price > 1d EMA50 AND volume > 1.8x 20-period average volume
# Short when price breaks below R1 (resistance) in ranging market AND price < 1d EMA50 AND volume > 1.8x 20-period average volume
# Uses 12h timeframe to reduce trade frequency, targeting 50-150 total trades over 4 years
# Camarilla pivots from daily timeframe provide precise intraday support/resistance levels
# EMA50 filter ensures trades align with higher timeframe trend
# Volume confirmation adds conviction to breakouts

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d EMA50 (trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 12h OHLC for Camarilla pivot calculation ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels using previous 12h bar's OHLC
    # Camarilla formulas: 
    # R4 = close + 1.5*(high-low)
    # R3 = close + 1.1*(high-low)
    # R2 = close + 0.6*(high-low)
    # R1 = close + 0.3*(high-low)
    # S1 = close - 0.3*(high-low)
    # S2 = close - 0.6*(high-low)
    # S3 = close - 1.1*(high-low)
    # S4 = close - 1.5*(high-low)
    hl_range = high_12h - low_12h
    camarilla_r1 = close_12h + 0.3 * hl_range
    camarilla_s1 = close_12h - 0.3 * hl_range
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1)
    
    # === 12h Volume Confirmation (20-period average) ===
    vol_12h = df_12h['volume'].values
    vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_50_val = ema_50_1d_aligned[i]
        r1_level = camarilla_r1_aligned[i]
        s1_level = camarilla_s1_aligned[i]
        vol_confirm = volume[i] > vol_ma_aligned[i] * 1.8  # 1.8x average volume for confirmation
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: price breaks above S1 (support bounce) AND price > 1d EMA50 AND volume confirmation
            # This captures mean reversion from support in ranging markets with trend alignment
            if price > s1_level and price > ema_50_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                continue
            # Short when: price breaks below R1 (resistance rejection) AND price < 1d EMA50 AND volume confirmation
            # This captures rejection at resistance in ranging markets with trend alignment
            elif price < r1_level and price < ema_50_val and vol_confirm:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions: reverse signal when opposite condition met
        elif position == 1:
            # Exit long if price breaks below S1 or goes below EMA50
            if price < s1_level or price < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short if price breaks above R1 or goes above EMA50
            if price > r1_level or price > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_1dEMA50_Volume1.8x"
timeframe = "12h"
leverage = 1.0