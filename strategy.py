#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot (H3/L3) breakout with 1d EMA50 trend filter and volume spike confirmation
# Camarilla levels derived from 1d OHLC provide institutional support/resistance zones
# Breakout above H3 (long) or below L3 (short) with volume > 2.0x 20-period EMA confirms momentum
# 1d EMA50 ensures trades align with higher timeframe trend, reducing whipsaws in ranging markets
# Designed for low trade frequency: ~20-35 trades/year per symbol with 0.30 sizing
# Works in bull/bear markets by following 1d trend direction via EMA50 filter

name = "4h_Camarilla_H3L3_Breakout_1dEMA50_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for trend filter (EMA50) and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d OHLC for Camarilla levels (H3, L3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    daily_range = high_1d - low_1d
    camarilla_h3 = close_1d + 1.1 * daily_range / 6  # H3 level
    camarilla_l3 = close_1d - 1.1 * daily_range / 6  # L3 level
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA (strict to reduce trades)
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(50, 20)  # Need 1d EMA50 and volume EMA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade in direction of 1d EMA50
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if uptrend:
                # Long: breakout above H3 with volume spike
                if close[i] > camarilla_h3_aligned[i] and volume_spike[i]:
                    signals[i] = 0.30
                    position = 1
                else:
                    signals[i] = 0.0
            elif downtrend:
                # Short: breakdown below L3 with volume spike
                if close[i] < camarilla_l3_aligned[i] and volume_spike[i]:
                    signals[i] = -0.30
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid sideways markets
        
        elif position == 1:  # Long position
            # Exit: price retests Camarilla H3 level (mean reversion at pivot)
            if close[i] <= camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: price retests Camarilla L3 level (mean reversion at pivot)
            if close[i] >= camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals