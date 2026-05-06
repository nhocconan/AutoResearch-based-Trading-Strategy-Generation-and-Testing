#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike
# Long when price breaks above Camarilla R3 level AND price > 4h EMA50 AND volume > 1.8 * 20-period avg volume
# Short when price breaks below Camarilla S3 level AND price < 4h EMA50 AND volume > 1.8 * 20-period avg volume
# Exit when price crosses Camarilla pivot point (mean reversion to equilibrium)
# Uses discrete sizing 0.25 to balance profit potential and drawdown control
# Target: 80-180 total trades over 4 years (20-45/year) for 4h timeframe
# Camarilla levels provide intraday structure, EMA50 filters trend direction, volume confirms participation

name = "4h_CamarillaR3S3_Breakout_4hEMA50_Trend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data ONCE before loop for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for today (based on yesterday's OHLC)
    # Camarilla formulas: 
    # H4 = Close + 1.5 * (High - Low)
    # H3 = Close + 1.125 * (High - Low)  <- R3 (resistance)
    # H2 = Close + 0.75 * (High - Low)
    # H1 = Close + 0.5 * (High - Low)
    # Pivot = (High + Low + Close) / 3
    # L1 = Close - 0.5 * (High - Low)
    # L2 = Close - 0.75 * (High - Low)
    # L3 = Close - 1.125 * (High - Low)  <- S3 (support)
    # L4 = Close - 1.5 * (High - Low)
    daily_range = high_1d - low_1d
    camarilla_h3 = close_1d + 1.125 * daily_range  # R3
    camarilla_l3 = close_1d - 1.125 * daily_range  # S3
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3.0  # Pivot point
    
    # Get 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50
    close_4h_series = pd.Series(close_4h)
    ema_50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume confirmation: volume > 1.8 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * avg_volume_20)
    
    # Align HTF indicators to 4h timeframe (wait for completed HTF bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3 with uptrend and volume spike
            if (close[i] > camarilla_h3_aligned[i] and close[i-1] <= camarilla_h3_aligned[i-1] and 
                close[i] > ema_50_4h_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 with downtrend and volume spike
            elif (close[i] < camarilla_l3_aligned[i] and close[i-1] >= camarilla_l3_aligned[i-1] and 
                  close[i] < ema_50_4h_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Camarilla pivot (mean reversion)
            if close[i] < camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Camarilla pivot (mean reversion)
            if close[i] > camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals