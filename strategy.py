#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla H3/L3 breakout with 12h volume regime filter and ATR stop
# Long when price breaks above H3 (bullish breakout) AND 12h volume > 1.5 * 24-bar avg volume (regime filter)
# Short when price breaks below L3 (bearish breakdown) AND 12h volume > 1.5 * 24-bar avg volume
# Exit with signal=0 when price reverts to the 12h VWAP (mean reversion to institutional value area)
# Uses discrete sizing 0.25 to limit fee drag and manage drawdown
# Target: 80-160 total trades over 4 years (20-40/year) for 4h timeframe
# H3/L3 levels are stronger than R3/S3 for breakouts with institutional follow-through
# 12h volume regime filter ensures we only trade during high-participation moves
# 12h VWAP exit provides adaptive mean reversion target that works in both bull and bear markets

name = "4h_Camarilla_H3L3_12hVolRegime_VWAPExit_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for volume regime filter and VWAP
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 24:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h VWAP (volume-weighted average price)
    typical_price_12h = (high_12h + low_12h + close_12h) / 3.0
    pv_12h = typical_price_12h * volume_12h
    cum_pv_12h = np.nancumsum(pv_12h)
    cum_vol_12h = np.nancumsum(volume_12h)
    vwap_12h = np.divide(cum_pv_12h, cum_vol_12h, out=np.full_like(cum_pv_12h, np.nan), where=cum_vol_12h!=0)
    
    # Calculate 12h volume regime: volume > 1.5 * 24-bar average volume
    avg_volume_24 = pd.Series(volume_12h).rolling(window=24, min_periods=24).mean().values
    volume_regime = volume_12h > (1.5 * avg_volume_24)
    
    # Align HTF indicators to 4h timeframe (wait for completed HTF bar)
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h)
    volume_regime_aligned = align_htf_to_ltf(prices, df_12h, volume_regime)
    
    # Get 1d data for Camarilla levels (more stable than intraday)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formula: H-L range based
    # H3/L3 are the key levels for breakout/breakdown with institutional relevance
    H3 = close_1d + (high_1d - low_1d) * 1.1 / 6
    L3 = close_1d - (high_1d - low_1d) * 1.1 / 6
    
    # Align Camarilla levels to 4h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(vwap_12h_aligned[i]) or np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(volume_regime_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Camarilla H3/L3 breakout/breakdown signals with volume regime filter
            # Long: price breaks above H3 (bullish breakout) AND volume regime active
            if close[i] > H3_aligned[i] and volume_regime_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 (bearish breakdown) AND volume regime active
            elif close[i] < L3_aligned[i] and volume_regime_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reverts to 12h VWAP (mean reversion to value area)
            if close[i] >= vwap_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reverts to 12h VWAP (mean reversion to value area)
            if close[i] <= vwap_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals