#!/usr/bin/env python3
"""
1h Momentum Breakout with 4h Trend Filter and Volume Spike
Hypothesis: On 1h timeframe, enter long when price breaks above 4h EMA50 with volume spike,
and enter short when price breaks below 4h EMA50 with volume spike. Use 1d ADX > 25 to filter
trending markets and avoid whipsaws in ranging conditions. Designed for 15-30 trades/year on 1h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA50 trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    ema_50 = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Get 1d data for ADX trend strength filter
    df_1d = get_htf_data(prices, '1d')
    # Calculate ADX(14) on daily data
    plus_dm = np.zeros(len(df_1d))
    minus_dm = np.zeros(len(df_1d))
    tr = np.zeros(len(df_1d))
    
    for i in range(1, len(df_1d)):
        high_diff = df_1d['high'].iloc[i] - df_1d['high'].iloc[i-1]
        low_diff = df_1d['low'].iloc[i-1] - df_1d['low'].iloc[i]
        plus_dm[i] = max(high_diff, 0) if high_diff > low_diff else 0
        minus_dm[i] = max(low_diff, 0) if low_diff > high_diff else 0
        tr[i] = max(
            df_1d['high'].iloc[i] - df_1d['low'].iloc[i],
            abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]),
            abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1])
        )
    
    # Smooth the values
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False).mean().values / atr_1d
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike: 2x 20-period average on 1h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_50_val = ema_50_aligned[i]
        adx_val = adx_aligned[i]
        
        if position == 0:
            # Only trade in trending markets (ADX > 25)
            if adx_val > 25:
                # Long: price breaks above 4h EMA50 with volume spike
                if price > ema_50_val and volume_spike[i]:
                    signals[i] = 0.20
                    position = 1
                # Short: price breaks below 4h EMA50 with volume spike
                elif price < ema_50_val and volume_spike[i]:
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.20
            # Exit: price returns below 4h EMA50
            if price < ema_50_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.20
            # Exit: price returns above 4h EMA50
            if price > ema_50_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_EMA50_Breakout_VolumeSpike_ADXFilter"
timeframe = "1h"
leverage = 1.0