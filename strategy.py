#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d EMA trend filter and volume confirmation
# - Primary: 12h price breaks above/below Camarilla H3/L3 levels from prior 1d
# - HTF: 1d EMA(50) trend filter (price > EMA50 for longs, < EMA50 for shorts)
# - Volume: 1d volume > 1.5x 20-period MA for conviction
# - Exit: Close crosses back inside H3/L3 levels
# - Position sizing: 0.25 discrete level
# - Works in bull/bear: EMA filter avoids counter-trend trades, volume confirms conviction, Camarilla adapts to volatility
# - Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe

name = "12h_1d_camarilla_breakout_ema_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 55:
        return np.zeros(n)
    
    # Pre-compute arrays
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla pivot levels (H3, L3) from prior day
    camarilla_h3 = np.full(len(close_1d), np.nan)
    camarilla_l3 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        if not (np.isnan(high_1d[i-1]) or np.isnan(low_1d[i-1]) or np.isnan(close_1d[i-1])):
            pp = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3.0
            range_ = high_1d[i-1] - low_1d[i-1]
            camarilla_h3[i] = pp + range_ * 1.1 / 4.0
            camarilla_l3[i] = pp - range_ * 1.1 / 4.0
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 / 51) + (ema_50_1d[i-1] * 49 / 51)
    
    # Calculate 1d volume MA(20) for volume confirmation
    volume_ma_20_1d = np.full(len(volume_1d), np.nan)
    for i in range(19, len(volume_1d)):
        if not np.isnan(volume_1d[i-19:i+1]).any():
            volume_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align HTF indicators to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(55, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i]) or
            np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirm = volume_1d_aligned[i] > 1.5 * volume_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long entry: Close > H3 + price > EMA50 + volume confirmation
            if close_12h[i] > camarilla_h3_aligned[i] and close_12h[i] > ema_50_1d_aligned[i] and volume_confirm:
                position = 1
                signals[i] = 0.25
            # Short entry: Close < L3 + price < EMA50 + volume confirmation
            elif close_12h[i] < camarilla_l3_aligned[i] and close_12h[i] < ema_50_1d_aligned[i] and volume_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:
            # Exit: Close crosses back inside H3/L3 levels
            if position == 1:
                if close_12h[i] < camarilla_h3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:
                if close_12h[i] > camarilla_l3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals