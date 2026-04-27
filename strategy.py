#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_Dyn
Hypothesis: Camarilla R3/S3 levels from 1d act as strong support/resistance. 
Breakout above R3 or below S3 with volume spike and 1d EMA34 trend filter captures 
institutional breakouts. Works in bull via R3 breaks above uptrend EMA, 
in bear via S3 breaks below downtrend EMA. Target: 25-40 trades/year.
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
    
    # Get 1d data for Camarilla calculation and EMA filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels (R3, S3) from prior 1d candle
    # Using prior day's OHLC to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior day's values (shifted by 1 to avoid look-ahead)
    phigh = np.roll(high_1d, 1)
    plow = np.roll(low_1d, 1)
    pclose = np.roll(close_1d, 1)
    phigh[0] = np.nan
    plow[0] = np.nan
    pclose[0] = np.nan
    
    # Camarilla calculations
    range_ = phigh - plow
    r3 = pclose + range_ * 1.1 / 4
    s3 = pclose - range_ * 1.1 / 4
    
    # Align to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike: volume > 2.0 * 20-period average (higher threshold for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for volume MA and Camarilla
    start_idx = 20  # for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: break above R3 with volume spike and price above EMA34 (uptrend)
            if close[i] > r3_val and vol_spike_val and close[i] > ema_34_val:
                signals[i] = size
                position = 1
            # Short: break below S3 with volume spike and price below EMA34 (downtrend)
            elif close[i] < s3_val and vol_spike_val and close[i] < ema_34_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price closes below R3 (failed breakout) or below EMA34 (trend change)
            if close[i] < r3_val or close[i] < ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price closes above S3 (failed breakdown) or above EMA34 (trend change)
            if close[i] > s3_val or close[i] > ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_Dyn"
timeframe = "4h"
leverage = 1.0