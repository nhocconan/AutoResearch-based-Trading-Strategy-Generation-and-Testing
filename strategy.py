#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels with 1d trend filter and volume confirmation.
# Uses Camarilla pivot levels (support/resistance) from daily timeframe for entry/exit.
# In trending markets, price tends to respect these levels as support/resistance.
# Combined with 1d EMA trend filter and volume spikes to avoid false breakouts.
# Target: 15-35 trades per year (60-140 total over 4 years) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for Camarilla pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day's OHLC
    # Using yesterday's data to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla formulas
    # Pivot = (High + Low + Close) / 3
    # R4 = Close + (High - Low) * 1.1/2
    # R3 = Close + (High - Low) * 1.1/4
    # R2 = Close + (High - Low) * 1.1/6
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    # S2 = Close - (High - Low) * 1.1/6
    # S3 = Close - (High - Low) * 1.1/4
    # S4 = Close - (High - Low) * 1.1/2
    
    pivot = np.full(len(prev_close), np.nan)
    r4 = np.full(len(prev_close), np.nan)
    r3 = np.full(len(prev_close), np.nan)
    r2 = np.full(len(prev_close), np.nan)
    r1 = np.full(len(prev_close), np.nan)
    s1 = np.full(len(prev_close), np.nan)
    s2 = np.full(len(prev_close), np.nan)
    s3 = np.full(len(prev_close), np.nan)
    s4 = np.full(len(prev_close), np.nan)
    
    for i in range(1, len(prev_close)):
        if not (np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(prev_close[i])):
            hl_range = prev_high[i] - prev_low[i]
            pivot[i] = (prev_high[i] + prev_low[i] + prev_close[i]) / 3.0
            r4[i] = prev_close[i] + hl_range * 1.1 / 2.0
            r3[i] = prev_close[i] + hl_range * 1.1 / 4.0
            r2[i] = prev_close[i] + hl_range * 1.1 / 6.0
            r1[i] = prev_close[i] + hl_range * 1.1 / 12.0
            s1[i] = prev_close[i] - hl_range * 1.1 / 12.0
            s2[i] = prev_close[i] - hl_range * 1.1 / 6.0
            s3[i] = prev_close[i] - hl_range * 1.1 / 4.0
            s4[i] = prev_close[i] - hl_range * 1.1 / 2.0
    
    # Align Camarilla levels to 12h timeframe
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot)
    r4_12h = align_htf_to_ltf(prices, df_1d, r4)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    s4_12h = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1-day EMA for trend filter
    ema20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Average volume (24-period = 12 hours) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(24, n):
        avg_volume[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(24, n):
        # Skip if any required data is not ready
        if (np.isnan(pivot_12h[i]) or np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or
            np.isnan(ema20_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_trend = ema20_1d_aligned[i]
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirm = vol > 2.0 * avg_vol
        
        if position == 0:
            # Long: Price breaks above R1 with volume + above EMA trend
            if (price > r1_12h[i] and volume_confirm and price > ema_trend):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below S1 with volume + below EMA trend
            elif (price < s1_12h[i] and volume_confirm and price < ema_trend):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price breaks below pivot or trend changes
            if (price < pivot_12h[i] or price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Price breaks above pivot or trend changes
            if (price > pivot_12h[i] or price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Camarilla_Pivot_Trend_Volume"
timeframe = "12h"
leverage = 1.0