#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation
# Camarilla pivots from 1d provide key intraday support/resistance levels. Breakout above R3
# or below S3 with 1w trend alignment and volume spike indicates strong momentum continuation.
# Works in bull/bear markets by following the higher timeframe trend.
# Target: 80-120 total trades over 4 years (20-30/year) with discrete sizing 0.25.

name = "6h_Camarilla_R3S3_Breakout_1wEMA50_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # Typical price = (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_ = df_1d['high'] - df_1d['low']
    
    # Camarilla levels
    camarilla_h5 = typical_price + 1.1 * range_ * 1.1 / 2  # R4
    camarilla_h4 = typical_price + 1.1 * range_ * 1.1 / 4  # R3
    camarilla_l4 = typical_price - 1.1 * range_ * 1.1 / 4  # S3
    camarilla_l5 = typical_price - 1.1 * range_ * 1.1 / 2  # S4
    
    # Align Camarilla levels to 6h timeframe (use previous day's levels)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4.values)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4.values)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1w for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation (2.0x 20-period average) on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for calculations)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Close breaks above camarilla H4 (R3) + 1w uptrend + volume spike
            if close[i] > camarilla_h4_aligned[i] and close[i-1] <= camarilla_h4_aligned[i-1] and close[i] > ema_50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: Close breaks below camarilla L4 (S3) + 1w downtrend + volume spike
            elif close[i] < camarilla_l4_aligned[i] and close[i-1] >= camarilla_l4_aligned[i-1] and close[i] < ema_50_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close breaks below camarilla L4 (S3) or trend reversal
            if close[i] < camarilla_l4_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close breaks above camarilla H4 (R3) or trend reversal
            if close[i] > camarilla_h4_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals