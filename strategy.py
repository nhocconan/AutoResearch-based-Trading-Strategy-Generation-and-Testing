#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA(34) trend filter and volume spike confirmation
# Camarilla pivot levels provide precise intraday support/resistance; breakout of R3/S3 indicates strong momentum
# 1d EMA(34) filters for primary trend alignment to avoid counter-trend trades
# Volume spike (2.0x 20-period average) confirms institutional participation
# Discrete position sizing 0.25 to minimize fee churn
# Targets 12-30 trades/year (50-120 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by requiring volume confirmation and primary trend alignment

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 12h Camarilla levels (based on previous day's high, low, close)
    # Need to align daily OHLC to 12h timeframe
    df_1d_ohlc = df_1d[['high', 'low', 'close']].copy()
    # Calculate Camarilla levels for each 1d bar
    h = df_1d_ohlc['high'].values
    l = df_1d_ohlc['low'].values
    c = df_1d_ohlc['close'].values
    # Camarilla formulas
    camarilla_r3 = c + (h - l) * 1.1 / 4
    camarilla_s3 = c - (h - l) * 1.1 / 4
    camarilla_r4 = c + (h - l) * 1.1 / 2
    camarilla_s4 = c - (h - l) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (wait for 1d bar to close)
    r3_12h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r4_12h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_12h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Camarilla calculation, EMA and volume MA)
    start_idx = 50  # buffer for indicators
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3 + price > 1d EMA + volume spike
            if close[i] > r3_12h[i] and close[i] > ema_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 + price < 1d EMA + volume spike
            elif close[i] < s3_12h[i] and close[i] < ema_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price retreats to Camarilla S3 level
            if close[i] < s3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises to Camarilla R3 level
            if close[i] > r3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals