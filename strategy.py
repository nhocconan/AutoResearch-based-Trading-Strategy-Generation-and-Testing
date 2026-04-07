#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian channel breakout with 1d Camarilla pivot confluence and volume confirmation
# Long when price breaks above 6h Donchian high (20) AND trades above 1d Camarilla H3 with volume > 1.5x average
# Short when price breaks below 6h Donchian low (20) AND trades below 1d Camarilla L3 with volume > 1.5x average
# Uses volume filter to avoid false breakouts and Camarilla levels for institutional support/resistance
# Designed for low trade frequency (12-37/year) to minimize fee drift while capturing strong trending moves
# Works in bull markets via breakout continuation and in bear markets via breakdown continuation

name = "6h_donchian20_1d_camarilla_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formula: based on previous day's range
    # H4 = close + 1.5*(high-low), H3 = close + 1.0*(high-low), L3 = close - 1.0*(high-low), L4 = close - 1.5*(high-low)
    camarilla_h3 = np.zeros(len(df_1d))
    camarilla_l3 = np.zeros(len(df_1d))
    
    for i in range(1, len(df_1d)):
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_val = prev_high - prev_low
        camarilla_h3[i] = prev_close + 1.0 * range_val
        camarilla_l3[i] = prev_close - 1.0 * range_val
    # First day has no previous data
    camarilla_h3[0] = camarilla_h3[1] if len(df_1d) > 1 else close_1d[0]
    camarilla_l3[0] = camarilla_l3[1] if len(df_1d) > 1 else close_1d[0]
    
    # Align Camarilla levels to 6h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume average (50-period)
    volume_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = volume[i] > 1.5 * volume_ma[i]
        
        # Long: Donchian breakout above H3 with volume
        if close[i] > donchian_high[i] and close[i] > camarilla_h3_aligned[i] and volume_confirm:
            signals[i] = 0.25
        # Short: Donchian breakdown below L3 with volume
        elif close[i] < donchian_low[i] and close[i] < camarilla_l3_aligned[i] and volume_confirm:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals