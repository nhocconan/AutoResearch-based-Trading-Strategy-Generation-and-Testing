#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 12h EMA50 trend filter and volume spike confirmation
# Long when price breaks above 4h Camarilla R4 level AND 12h close > 12h EMA50 AND volume > 2.0 * 20-bar average volume
# Short when price breaks below 4h Camarilla S4 level AND 12h close < 12h EMA50 AND volume > 2.0 * 20-bar average volume
# Exit when price retests the 4h Camarilla pivot point (PP)
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Camarilla R4/S4 levels provide stronger breakout confirmation than R3/S3
# 12h EMA50 filters for higher timeframe trend alignment (proven BTC/ETH edge)
# Volume spike confirmation reduces false breakouts during low participation
# Works in both bull and bear markets by following the 12h trend

name = "4h_Camarilla_R4S4_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h Camarilla levels and 12h EMA50 ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_4h) < 1 or len(df_12h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    close_12h = df_12h['close'].values
    
    # Calculate 4h Camarilla levels (based on previous bar's OHLC)
    # Camarilla: PP = (H+L+C)/3, R4 = C + (H-L)*1.1, S4 = C - (H-L)*1.1
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_pp = (prev_high + prev_low + prev_close) / 3.0
    camarilla_r4 = prev_close + ((prev_high - prev_low) * 1.1)
    camarilla_s4 = prev_close - ((prev_high - prev_low) * 1.1)
    
    # Calculate 12h EMA50 trend filter
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 4h timeframe (wait for completed bars)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pp)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s4)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: volume > 2.0 * 20-bar average volume (spike filter)
    volume_series = pd.Series(volume)
    avg_volume_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > R4 level AND uptrend AND volume spike
            if close[i] > camarilla_r4_aligned[i] and close[i] > ema50_12h_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < S4 level AND downtrend AND volume spike
            elif close[i] < camarilla_s4_aligned[i] and close[i] < ema50_12h_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests pivot point from above
            if close[i] <= camarilla_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests pivot point from below
            if close[i] >= camarilla_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals