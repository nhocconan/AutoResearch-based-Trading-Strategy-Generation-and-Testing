#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Camarilla pivot levels with volume confirmation
# Fade at R3/S3 levels (mean reversion) and breakout continuation at R4/S4 levels (trend following)
# Uses 1d ADX to filter regime: ADX > 25 for breakout mode, ADX < 20 for mean reversion mode
# Discrete position sizing 0.25 to target ~25-40 trades/year and minimize fee drag
# Works in bull/bear markets: adapts to regime via ADX filter

name = "6h_12h_1d_camarilla_adaptive_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla levels (based on prior 12h bar)
    camarilla_h4 = close_12h + 1.5 * (high_12h - low_12h)
    camarilla_h3 = close_12h + 1.25 * (high_12h - low_12h)
    camarilla_l3 = close_12h - 1.25 * (high_12h - low_12h)
    camarilla_l4 = close_12h - 1.5 * (high_12h - low_12h)
    
    # Align 12h Camarilla levels to 6h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    
    # Load 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.maximum(np.abs(high_1d[1:] - high_1d[:-1]), 
                               np.abs(low_1d[1:] - low_1d[:-1])))
    # Pad first element
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    # Wilder's smoothing
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute volume confirmation (20-period 6h volume MA)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 6h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit conditions based on regime
            if adx_aligned[i] > 25:  # Trending regime - breakout mode
                # Exit long if price falls below L3 (mean reversion exit)
                if close[i] < camarilla_l3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # Ranging regime - mean reversion mode
                # Exit long if price reaches L4 (profit target) or goes above H3 (failure)
                if close[i] >= camarilla_h3_aligned[i] or close[i] <= camarilla_l4_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit conditions based on regime
            if adx_aligned[i] > 25:  # Trending regime - breakout mode
                # Exit short if price rises above H3 (mean reversion exit)
                if close[i] > camarilla_h3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # Ranging regime - mean reversion mode
                # Exit short if price reaches H4 (profit target) or goes below L3 (failure)
                if close[i] <= camarilla_l3_aligned[i] or close[i] >= camarilla_h4_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            # Entry logic based on regime
            if adx_aligned[i] > 25:  # Trending regime - breakout mode
                # Breakout continuation: enter on break of H4/L4 with volume
                if close[i] > camarilla_h4_aligned[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < camarilla_l4_aligned[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
            else:  # Ranging regime - mean reversion mode
                # Mean reversion: enter on touch of H3/L3 with volume
                if close[i] < camarilla_l3_aligned[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                elif close[i] > camarilla_h3_aligned[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
    
    return signals