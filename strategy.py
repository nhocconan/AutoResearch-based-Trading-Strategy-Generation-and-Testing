#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with weekly trend filter and volume confirmation
# - Long: price breaks above Camarilla R4 (1d) + 1w close > 1w EMA20 (uptrend) + volume > 2.0x 24-period average
# - Short: price breaks below Camarilla S4 (1d) + 1w close < 1w EMA20 (downtrend) + volume > 2.0x 24-period average
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Designed for 6h timeframe: targets 12-37 trades/year to avoid fee drag
# - Works in bull/bear markets: weekly trend filter prevents counter-trend trades, Camarilla breakouts capture momentum

name = "6h_1d_1w_camarilla_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla levels (based on previous day OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Camarilla levels: based on previous day's range
    rng = high_1d - low_1d
    camarilla_h4 = close_1d + 1.1/2 * rng  # R4
    camarilla_l4 = close_1d - 1.1/2 * rng  # S4
    camarilla_h3 = close_1d + 1.1/4 * rng  # R3
    camarilla_l3 = close_1d - 1.1/4 * rng  # S3
    
    # Align Camarilla levels to 6h timeframe (use previous day's levels)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Pre-compute 1w EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Pre-compute 6h volume confirmation
    volume_6h = prices['volume'].values
    avg_volume_24 = pd.Series(volume_6h).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume_6h > (2.0 * avg_volume_24)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Camarilla L3 (profit target or reversal)
            if low_6h[i] < camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Camarilla H3 (profit target or reversal)
            if high_6h[i] > camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakout with trend and volume filters
            if vol_spike[i]:
                # Long: price breaks above Camarilla R4 + 1w uptrend (close > EMA20)
                if high_6h[i] > camarilla_h4_aligned[i] and close_6h[i] > ema_20_1w_aligned[i]:
                    position = 1
                    entry_price = close_6h[i]
                    signals[i] = 0.25
                # Short: price breaks below Camarilla S4 + 1w downtrend (close < EMA20)
                elif low_6h[i] < camarilla_l4_aligned[i] and close_6h[i] < ema_20_1w_aligned[i]:
                    position = -1
                    entry_price = close_6h[i]
                    signals[i] = -0.25
    
    return signals