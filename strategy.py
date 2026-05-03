#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and 1d volume confirmation.
# Uses Camarilla levels (R1/S1) from prior 4h bar for breakout entries.
# Trend filter: 4h EMA50 slope > 0 for longs, < 0 for shorts.
# Volume filter: 1h volume > 1.5x 20-period MA on 1d timeframe (institutional participation).
# Session filter: 08-20 UTC to avoid low-liquidity hours.
# Discrete sizing: 0.20 to control risk and minimize fee churn.
# Target: 60-150 total trades over 4 years (15-37/year) with Sharpe > 0 on BTC/ETH/SOL.

name = "1h_Camarilla_R1S1_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-compute session hours (UTC) - open_time is already datetime64[ms]
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla levels and trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend (using close)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_slope_4h = np.diff(ema_50_4h, prepend=ema_50_4h[0])  # slope = current - previous
    
    # Calculate Camarilla levels from prior 4h bar (HLC of completed bar)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    # Camarilla R1, S1 based on prior bar's range
    rango = high_4h - low_4h
    camarilla_r1 = close_4h + (rango * 1.1 / 12)
    camarilla_s1 = close_4h - (rango * 1.1 / 12)
    
    # Align 4h indicators to 1h timeframe (wait for completed 4h bar)
    ema_50_slope_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_slope_4h)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Calculate 1d volume MA20 for institutional participation filter
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * vol_ma_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # 1h volume for entry timing (optional confirmation)
    volume_1h = prices['volume'].values
    vol_ma_20_1h = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    vol_spike_1h = volume_1h > (1.5 * vol_ma_20_1h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup for EMA50
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = prices['close'].iloc[i]
        
        # Get aligned values
        slope = ema_50_slope_4h_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        vol_spike = vol_spike_1d_aligned[i] > 0.5  # boolean from aligned float
        
        # Skip if any value is NaN
        if np.isnan(slope) or np.isnan(r1) or np.isnan(s1):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Entry logic
        if position == 0:
            # Long: price breaks above R1 AND uptrend (slope > 0) AND volume spike
            if close_val > r1 and slope > 0 and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 AND downtrend (slope < 0) AND volume spike
            elif close_val < s1 and slope < 0 and vol_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price closes below S1 OR trend reverses (slope < 0)
            if close_val < s1 or slope < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price closes above R1 OR trend reverses (slope > 0)
            if close_val > r1 or slope > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals