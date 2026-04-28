#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d Camarilla pivot R1/S1 breakout with volume spike and 4h EMA50 trend filter.
# Enter long when price breaks above 4h Camarilla R1 with volume spike and above 4h EMA50.
# Enter short when price breaks below 4h Camarilla S1 with volume spike and below 4h EMA50.
# Uses discrete position sizing (0.20) to minimize fee drag. Target: 15-37 trades/year on 1h.
# Session filter (08-20 UTC) reduces noise trades. Works in bull (breakouts with trend) and bear (failed breaks reverse).

name = "1h_Camarilla_R1S1_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla pivots and EMA50 (HTF)
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h Camarilla pivots (using previous bar's high, low, close)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    n_4h = len(high_4h)
    camarilla_r1 = np.full(n_4h, np.nan)
    camarilla_s1 = np.full(n_4h, np.nan)
    
    for i in range(1, n_4h):
        # Use previous bar to avoid look-ahead
        phigh = high_4h[i-1]
        plow = low_4h[i-1]
        pclose = close_4h[i-1]
        pivot = (phigh + plow + pclose) / 3.0
        rng = phigh - plow
        camarilla_r1[i] = pivot + rng * 1.1 / 4.0
        camarilla_s1[i] = pivot - rng * 1.1 / 4.0
    
    # Forward fill Camarilla levels
    camarilla_r1 = pd.Series(camarilla_r1).ffill().values
    camarilla_s1 = pd.Series(camarilla_s1).ffill().values
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h indicators to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h volume spike: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 4h EMA50
        above_ema = close[i] > ema_50_4h_aligned[i]
        below_ema = close[i] < ema_50_4h_aligned[i]
        
        # Camarilla breakout conditions with volume confirmation
        long_breakout = close[i] > camarilla_r1_aligned[i] and volume_spike[i]
        short_breakout = close[i] < camarilla_s1_aligned[i] and volume_spike[i]
        
        # Exit conditions: opposite Camarilla level or trend reversal
        long_exit = close[i] < camarilla_s1_aligned[i] or below_ema
        short_exit = close[i] > camarilla_r1_aligned[i] or above_ema
        
        # Handle entries and exits
        if long_breakout and above_ema and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_breakout and below_ema and position >= 0:
            signals[i] = -0.20
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals