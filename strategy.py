#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with volume confirmation and 1d EMA50 filter
# Uses 12h primary timeframe with 1d HTF for trend alignment and weekly Camarilla pivots for structure.
# Volume confirmation filters breakouts, EMA50 ensures trend alignment. Discrete position sizing (0.25) minimizes fee churn.
# Target: 50-150 trades over 4 years (12-37/year) to avoid fee drag while maintaining statistical significance.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h data (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # === 1d data (higher timeframe for trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # === 1w data (higher timeframe for Camarilla pivot calculation) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === 1d EMA50 for trend filter ===
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === Weekly Camarilla pivot levels (R1, S1) ===
    # Camarilla formula: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1_1w = close_1w + (high_1w - low_1w) * 1.1 / 12
    camarilla_s1_1w = close_1w - (high_1w - low_1w) * 1.1 / 12
    
    # Align weekly Camarilla levels to 12h timeframe with 1-bar delay (wait for weekly close)
    camarilla_r1_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1_1w)
    camarilla_s1_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1_1w)
    
    # === 12h volume confirmation ===
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume_12h > (1.5 * vol_ma_20_12h)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(camarilla_r1_1w_aligned[i]) or
            np.isnan(camarilla_s1_1w_aligned[i]) or
            np.isnan(vol_ma_20_12h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r1 = camarilla_r1_1w_aligned[i]
        s1 = camarilla_s1_1w_aligned[i]
        ema50 = ema50_1d_aligned[i]
        vol_conf = vol_confirm[i]
        
        # === STOPLOSS LOGIC (ATR-based) ===
        if position == 1:  # Long position
            atr_12h = np.abs(high_12h - low_12h)
            atr_ma = pd.Series(atr_12h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_12h, atr_ma)
            atr_val = atr_aligned[i]
            if price < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            atr_12h = np.abs(high_12h - low_12h)
            atr_ma = pd.Series(atr_12h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_12h, atr_ma)
            atr_val = atr_aligned[i]
            if price > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price reaches weekly R4 or shows weakness
            camarilla_r4_1w = close_1w + (high_1w - low_1w) * 1.1/2
            camarilla_r4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4_1w)
            if not np.isnan(camarilla_r4_1w_aligned[i]) and price >= camarilla_r4_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price reaches weekly S4
            camarilla_s4_1w = close_1w - (high_1w - low_1w) * 1.1/2
            camarilla_s4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4_1w)
            if not np.isnan(camarilla_s4_1w_aligned[i]) and price <= camarilla_s4_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require volume confirmation and trend alignment
            if vol_conf:
                # Go long when price breaks above weekly R1 and above 1d EMA50 (bullish alignment)
                if price > r1 and price > ema50:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Go short when price breaks below weekly S1 and below 1d EMA50 (bearish alignment)
                elif price < s1 and price < ema50:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_Volume_EMA50_Filter"
timeframe = "12h"
leverage = 1.0