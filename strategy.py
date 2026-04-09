#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout + 4h/1d trend filter + volume confirmation
# - Primary signal: 1h price breaks above/below Camarilla pivot levels (H3/L3) from prior day
# - Trend filter: 4h EMA50 and 1d EMA200 - price must be above both for longs, below for shorts
# - Volume confirmation: 1h volume > 24-period median volume (avoid low-participation signals)
# - Session filter: 08-20 UTC (reduce noise, focus on liquid London/NY overlap)
# - Position size: 0.20 (discrete level) to minimize fee churn
# - Target: 15-37 trades/year (60-150 total over 4 years) per 1h strategy guidelines
# - Works in bull/bear: Camarilla pivots act as dynamic support/resistance, EMA filters ensure alignment with higher timeframe trend

name = "1h_4h_1d_camarilla_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Pre-compute 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Pre-compute prior day's Camarilla pivot levels (H3, L3) for 1h breakout
    # Using prior day's OHLC to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    
    # Align Camarilla levels to 1h timeframe (use prior day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 1h volume regime: volume > 24-period median volume (avoid low-participation signals)
    volume = prices['volume'].values
    median_volume_24 = pd.Series(volume).rolling(window=24, min_periods=24).median().values
    volume_regime = volume > median_volume_24
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_regime[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Camarilla H3 OR breaks below 4h EMA50
            if prices['close'].iloc[i] < camarilla_h3_aligned[i] or prices['close'].iloc[i] < ema_50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla L3 OR breaks above 4h EMA50
            if prices['close'].iloc[i] > camarilla_l3_aligned[i] or prices['close'].iloc[i] > ema_50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for Camarilla breakout with volume confirmation and trend filter
            # Long: price breaks above Camarilla H3 AND volume regime AND above both EMAs
            if (prices['close'].iloc[i] > camarilla_h3_aligned[i] and 
                volume_regime[i] and 
                prices['close'].iloc[i] > ema_50_4h_aligned[i] and 
                prices['close'].iloc[i] > ema_200_1d_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short: price breaks below Camarilla L3 AND volume regime AND below both EMAs
            elif (prices['close'].iloc[i] < camarilla_l3_aligned[i] and 
                  volume_regime[i] and 
                  prices['close'].iloc[i] < ema_50_4h_aligned[i] and 
                  prices['close'].iloc[i] < ema_200_1d_aligned[i]):
                position = -1
                signals[i] = -0.20
    
    return signals