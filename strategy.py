#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R4/S4 breakout with 1d EMA34 trend filter and volume spike
# Long when price breaks above Camarilla R4 level AND price > 1d EMA34 AND volume > 2.0 * 20-period avg volume
# Short when price breaks below Camarilla S4 level AND price < 1d EMA34 AND volume > 2.0 * 20-period avg volume
# Exit when price crosses Camarilla pivot point (mean reversion to equilibrium)
# Uses discrete sizing 0.25 to balance profit potential and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Weekly trend filter (1w EMA200) added to avoid counter-trend trades in strong weekly trends
# Camarilla R4/S4 represent strong breakout levels, reducing false signals vs R3/S3
# Volume spike threshold increased to 2.0x for stronger confirmation

name = "6h_CamarillaR4S4_Breakout_1dEMA34_1wEMA200_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data ONCE before loop for Camarilla pivot levels and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for today (based on yesterday's OHLC)
    daily_range = high_1d - low_1d
    camarilla_h4 = close_1d + 1.5 * daily_range  # R4
    camarilla_l4 = close_1d - 1.5 * daily_range  # S4
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3.0  # Pivot point
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get weekly data ONCE before loop for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 210:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA200
    close_1w_series = pd.Series(close_1w)
    ema_200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    # Align HTF indicators to 6h timeframe (wait for completed HTF bar)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R4 with uptrend (1d and 1w) and volume spike
            if (close[i] > camarilla_h4_aligned[i] and close[i-1] <= camarilla_h4_aligned[i-1] and 
                close[i] > ema_34_1d_aligned[i] and close[i] > ema_200_1w_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S4 with downtrend (1d and 1w) and volume spike
            elif (close[i] < camarilla_l4_aligned[i] and close[i-1] >= camarilla_l4_aligned[i-1] and 
                  close[i] < ema_34_1d_aligned[i] and close[i] < ema_200_1w_aligned[i] and volume_spike[i]):
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