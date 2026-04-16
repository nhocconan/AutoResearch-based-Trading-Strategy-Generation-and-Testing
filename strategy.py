#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R for mean reversion in ranging markets and 1d ADX for trend filtering.
# Long when Williams %R < -80 (oversold) AND ADX < 25 (ranging market) AND price > 1d VWAP (bullish bias).
# Short when Williams %R > -20 (overbought) AND ADX < 25 (ranging market) AND price < 1d VWAP (bearish bias).
# Uses discrete position size 0.25. Williams %R identifies extremes, ADX filters ranging conditions, VWAP provides intraday bias.
# Works in bull markets (mean reversion longs in ranges) and bear markets (mean reversion shorts in ranges).
# 6h timeframe targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data once before loop for Williams %R, ADX, and VWAP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Williams %R (14) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14) * -100
    
    # === 1d Indicators: ADX (14) ===
    # ADX calculation: +DM, -DM, TR, then smoothed
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    # Pad to same length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothed values (Wilder's smoothing)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # === 1d Indicators: VWAP ===
    typical_price = (high_1d + low_1d + close_1d) / 3
    vwap = np.cumsum(typical_price * volume_1d) / np.cumsum(volume_1d)
    
    # Align all indicators to primary timeframe (6h)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 30  # Williams %R and ADX need sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vwap_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        williams_r = williams_r_aligned[i]
        adx_val = adx_aligned[i]
        vwap = vwap_aligned[i]
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when Williams %R > -50 (exits oversold) OR ADX > 30 (trending market)
            if (williams_r > -50) or (adx_val > 30):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when Williams %R < -50 (exits overbought) OR ADX > 30 (trending market)
            if (williams_r < -50) or (adx_val > 30):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND ADX < 25 (ranging) AND price > VWAP (bullish bias)
            if (williams_r < -80) and (adx_val < 25) and (price > vwap):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Williams %R > -20 (overbought) AND ADX < 25 (ranging) AND price < VWAP (bearish bias)
            elif (williams_r > -20) and (adx_val < 25) and (price < vwap):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "6h_1dWilliamsR_ADX_VWAP_MeanReversion_V1"
timeframe = "6h"
leverage = 1.0