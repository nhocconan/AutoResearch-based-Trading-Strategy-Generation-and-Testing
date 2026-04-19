# Hypothesis: 4h timeframe with daily pivot-based breakout strategy focusing on R2/S2 levels (less extreme than R1/S1, more reliable than R4/S4) with volume confirmation and trend filter. Targets 20-40 trades/year to avoid fee drag while capturing meaningful breakouts in both bull and bear markets.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Pivot_R2S2_Breakout_Volume_Trend_v1"
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
    
    # Get daily data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points from previous day
    prev_close = np.roll(close_1d, 1)
    prev_close[0] = np.nan
    prev_high = np.roll(high_1d, 1)
    prev_high[0] = np.nan
    prev_low = np.roll(low_1d, 1)
    prev_low[0] = np.nan
    
    # Pivot = (H + L + C) / 3
    pivot = (prev_high + prev_low + prev_close) / 3.0
    # R2 = P + 2*(H - L)
    r2 = pivot + 2.0 * (prev_high - prev_low)
    # S2 = P - 2*(H - L)
    s2 = pivot - 2.0 * (prev_high - prev_low)
    # R4 = C + (H - L) * 1.1 / 2
    r4 = prev_close + (prev_high - prev_low) * 1.1 / 2.0
    # S4 = C - (H - L) * 1.1 / 2
    s4 = prev_close - (prev_high - prev_low) * 1.1 / 2.0
    
    # Align to 4h timeframe
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot)
    r2_4h = align_htf_to_ltf(prices, df_1d, r2)
    s2_4h = align_htf_to_ltf(prices, df_1d, s2)
    r4_4h = align_htf_to_ltf(prices, df_1d, r4)
    s4_4h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Trend filter: 50-period EMA on 4h close
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(pivot_4h[i]) or np.isnan(r2_4h[i]) or np.isnan(s2_4h[i]) or \
           np.isnan(r4_4h[i]) or np.isnan(s4_4h[i]) or np.isnan(ema_50[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 1.5x average
        volume_spike = vol > 1.5 * vol_ma
        
        # Trend condition: price above/below EMA50
        uptrend = price > ema_50[i]
        downtrend = price < ema_50[i]
        
        if position == 0:
            # Long: Price breaks above R2 with volume spike and uptrend
            if price > r2_4h[i] and volume_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S2 with volume spike and downtrend
            elif price < s2_4h[i] and volume_spike and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below S2 (reversal signal) or breaks S4 (strong reversal)
            if price < s2_4h[i] or price < s4_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above R2 (reversal signal) or breaks R4 (strong reversal)
            if price > r2_4h[i] or price > r4_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals