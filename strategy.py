#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses 4h timeframe for signal generation with Camarilla pivot levels from 1d data
# 1d EMA34 provides higher timeframe trend filter to avoid counter-trend trades
# Volume confirmation (2.0x 20-period average) ensures institutional participation
# Discrete position sizing (0.25) balances return and risk
# Target: 75-200 total trades over 4 years = 19-50/year for 4h timeframe
# Works in bull markets via trend-aligned breakouts, in bear via trend filter avoiding false signals
# Based on proven Camarilla breakout structure with volume confirmation and HTF trend filter

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_v1"
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
    
    # Load 4h data ONCE before loop for price action
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for Camarilla pivots and EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from 1d OHLC
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C = close, H = high, L = low of previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values  # same as close_1d for current bar
    
    # Calculate Camarilla levels using previous day's OHLC
    camarilla_R1 = close_1d_prev + (high_1d - low_1d) * 1.1 / 12
    camarilla_S1 = close_1d_prev - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_confirm[i]) or
            np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Camarilla R1 + price > 1d EMA34 + volume confirm
            if close[i] > camarilla_R1_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S1 + price < 1d EMA34 + volume confirm
            elif close[i] < camarilla_S1_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Camarilla S1 or reverse signal
            if close[i] < camarilla_S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Camarilla R1 or reverse signal
            if close[i] > camarilla_R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals