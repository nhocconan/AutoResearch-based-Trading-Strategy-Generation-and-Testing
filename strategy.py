#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and session timing
# - Uses 1h Camarilla pivot levels (H3/L3) for mean reversion entries in ranging markets
# - 4h EMA(50) trend filter: only trade in direction of 4h trend to avoid counter-trend whipsaws
# - Session filter: 08-20 UTC to avoid low-volume Asian session noise
# - Volume confirmation: 1h volume > 1.5x 20-period average to ensure breakout validity
# - Position size: 0.20 (20% of capital) - discrete level to minimize fee churn
# - Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe
# - Works in both bull/bear: Camarilla pivots capture mean reversion in ranges, 
#   4h trend filter allows trend continuation in strong moves, session filter reduces noise

name = "1h_4h_camarilla_trend_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 60 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h indicators
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Pre-compute 1d indicators for Camarilla pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Camarilla: H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4
    #            L3 = close - 1.1*(high-low)*1.1/4, L4 = close - 1.1*(high-low)*1.1/2
    rng_1d = high_1d - low_1d
    camarilla_h3 = close_1d + (1.1 * rng_1d * 1.1 / 4)
    camarilla_l3 = close_1d - (1.1 * rng_1d * 1.1 / 4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 1h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h volume > 1.5x 20-period average (volume confirmation)
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute hours for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_4h_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches Camarilla H3 or reverses below VWAP approximation
            if high[i] >= camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price reaches Camarilla L3 or reverses above VWAP approximation
            if low[i] <= camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for Camarilla breakout with 4h trend filter and volume confirmation
            # Long: price breaks above Camarilla H3 AND 4h EMA trending up AND volume spike
            if (high[i] >= camarilla_h3_aligned[i] and 
                close[i] > ema_4h_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.20
            # Short: price breaks below Camarilla L3 AND 4h EMA trending down AND volume spike
            elif (low[i] <= camarilla_l3_aligned[i] and 
                  close[i] < ema_4h_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.20
    
    return signals