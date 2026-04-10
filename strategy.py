#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Long when price breaks above Camarilla H4 resistance in 1w uptrend (close > EMA50) with volume spike
# - Short when price breaks below Camarilla L4 support in 1w downtrend (close < EMA50) with volume spike
# - Uses discrete position sizing (0.30) to minimize fee churn
# - Targets 12-37 trades/year (50-150 total over 4 years) to avoid fee drag
# - Weekly trend filter reduces false breakouts in ranging markets

name = "6h_1w_camarilla_breakout_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 50 or len(df_1d) < 100:
        return np.zeros(n)
    
    # Pre-compute 1w indicators
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1w Camarilla pivot levels (based on previous week)
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # H4 = close + 1.5*(high-low)*1.1/2
    # L4 = close - 1.5*(high-low)*1.1/2
    camarilla_h4 = close_1w + 1.5 * (high_1w - low_1w) * 1.1 / 2
    camarilla_l4 = close_1w - 1.5 * (high_1w - low_1w) * 1.1 / 2
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d volume confirmation: > 2.0x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (2.0 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Camarilla H3 (take profit) or reverses below L4 (stop)
            camarilla_h3 = close_1w + 1.1 * (high_1w - low_1w) * 1.1 / 4
            camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
            if (prices['close'].iloc[i] < camarilla_h3_aligned[i] or 
                prices['close'].iloc[i] < camarilla_l4_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla L3 (take profit) or reverses above H4 (stop)
            camarilla_l3 = close_1w - 1.1 * (high_1w - low_1w) * 1.1 / 4
            camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
            if (prices['close'].iloc[i] > camarilla_l3_aligned[i] or 
                prices['close'].iloc[i] > camarilla_h4_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Look for Camarilla H4/L4 breakout with trend and volume filters
            if vol_spike_1d_aligned[i]:
                # Long signal: price breaks above H4 resistance in 1w uptrend
                if (prices['high'].iloc[i] > camarilla_h4_aligned[i] and 
                    prices['close'].iloc[i] > ema_50_1w_aligned[i]):
                    position = 1
                    signals[i] = 0.30
                # Short signal: price breaks below L4 support in 1w downtrend
                elif (prices['low'].iloc[i] < camarilla_l4_aligned[i] and 
                      prices['close'].iloc[i] < ema_50_1w_aligned[i]):
                    position = -1
                    signals[i] = -0.30
    
    return signals