# 4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
# Strategy: Camarilla pivot breakout with 1d trend filter and volume confirmation
# Works in bull/bear: breakouts capture momentum, 1d trend avoids counter-trend trades, volume filters false breaks
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

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
    
    # Camarilla levels from previous day
    # R1 = close + (high - low) * 1.1 / 12
    # S1 = close - (high - low) * 1.1 / 12
    # Need previous day's high, low, close
    # We'll calculate daily OHLC first
    # Since we're on 4h timeframe, we can get 1d data and calculate levels per day
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    # R1 = C + (H-L) * 1.1/12
    # S1 = C - (H-L) * 1.1/12
    camarilla_R1 = df_1d['close'] + (df_1d['high'] - df_1d['low']) * 1.1 / 12
    camarilla_S1 = df_1d['close'] - (df_1d['high'] - df_1d['low']) * 1.1 / 12
    
    # Align to 4h timeframe (these levels are valid for the entire day after they're calculated)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1.values)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1.values)
    
    # Get 1d trend filter: EMA50
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + 1d uptrend + volume
            if close[i] > camarilla_R1_aligned[i] and close[i] > ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + 1d downtrend + volume
            elif close[i] < camarilla_S1_aligned[i] and close[i] < ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to the broken level (mean reversion)
            if position == 1:
                if close[i] <= camarilla_R1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] >= camarilla_S1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3