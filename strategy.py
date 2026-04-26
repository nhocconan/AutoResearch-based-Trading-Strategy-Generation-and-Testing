#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hEMA50_Trend_VolumeSpike_Session
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation (>1.8x 20-period MA).
Long when price breaks above R1 in 4h uptrend with volume spike. Short when price breaks below S1 in 4h downtrend with volume spike.
Uses session filter (08-20 UTC) to avoid low-liquidity hours. Discrete position sizing (0.20) to minimize fee churn.
Targets 15-37 trades/year on 1h timeframe by requiring confluence of HTF trend, volume spike, and session.
Designed to work in both bull and bear markets by following the 4h trend. Avoids false breakouts in choppy markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to reduce noise trades
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla pivot calculation and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 4h candle (avoid look-ahead)
    # Camarilla: R1 = C + (H-L)*1.0/6, S1 = C - (H-L)*1.0/6
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    camarilla_r1 = close_4h + (high_4h - low_4h) * 1.0 / 6
    camarilla_s1 = close_4h - (high_4h - low_4h) * 1.0 / 6
    
    # Align Camarilla levels to 1h timeframe (they change only when new 4h candle forms)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # 4h EMA50 trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    uptrend_4h = close > ema_50_4h_aligned
    downtrend_4h = close < ema_50_4h_aligned
    
    # Volume confirmation: volume > 1.8x 20-period MA (tight threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA, 50 for EMA)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_spike[i]) or
            not in_session[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        if position == 0:
            # Long: price breaks above R1 with 4h uptrend and volume spike
            if (close[i] > camarilla_r1_aligned[i] and 
                uptrend_4h[i] and volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 with 4h downtrend and volume spike
            elif (close[i] < camarilla_s1_aligned[i] and 
                  downtrend_4h[i] and volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: price closes below S1 (breakdown) OR 4h trend changes to downtrend
            if (close[i] < camarilla_s1_aligned[i] or not uptrend_4h[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: price closes above R1 (breakout) OR 4h trend changes to uptrend
            if (close[i] > camarilla_r1_aligned[i] or not downtrend_4h[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hEMA50_Trend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0