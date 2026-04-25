#!/usr/bin/env python3
"""
6h Camarilla R3/S3 Breakout + 1w EMA50 Trend + Volume Spike
Hypothesis: Camarilla R3/S3 levels act as strong intraday support/resistance. 
Breakouts beyond these levels with weekly EMA50 trend alignment and volume confirmation 
capture sustained momentum moves. Works in bull markets (breakouts with trend) and 
bear markets (failed breakouts/reversals at R3/S3). Targets 15-25 trades/year to 
minimize fee drag on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for EMA50 trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    ema_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # We use the previous day's OHLC to avoid look-ahead
    camarilla_R3 = np.full(n, np.nan)
    camarilla_S3 = np.full(n, np.nan)
    camarilla_R4 = np.full(n, np.nan)
    camarilla_S4 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Get previous 1d bar's OHLC (aligned to current 6h bar)
        # We need to find the 1d bar that closed before current 6h bar
        # Since we're using daily data aligned to 6h, we can use the aligned values
        # But for Camarilla we need the actual previous day's values
        # Simpler approach: calculate once per day and align
        pass  # We'll calculate differently below
    
    # Better approach: calculate Camarilla on 1d data, then align
    # Calculate Camarilla levels for each 1d bar
    camarilla_R3_1d = np.zeros(len(df_1d))
    camarilla_S3_1d = np.zeros(len(df_1d))
    camarilla_R4_1d = np.zeros(len(df_1d))
    camarilla_S4_1d = np.zeros(len(df_1d))
    
    for i in range(1, len(df_1d)):
        # Use previous day's OHLC to calculate today's levels (no look-ahead)
        H = df_1d['high'].iloc[i-1]
        L = df_1d['low'].iloc[i-1]
        C = df_1d['close'].iloc[i-1]
        
        camarilla_R4_1d[i] = C + ((H - L) * 1.1 / 2)
        camarilla_R3_1d[i] = C + ((H - L) * 1.1 / 4)
        camarilla_S3_1d[i] = C - ((H - L) * 1.1 / 4)
        camarilla_S4_1d[i] = C - ((H - L) * 1.1 / 2)
    
    # First bar has no previous day, so set to 0 or nan
    camarilla_R4_1d[0] = np.nan
    camarilla_R3_1d[0] = np.nan
    camarilla_S3_1d[0] = np.nan
    camarilla_S4_1d[0] = np.nan
    
    # Align Camarilla levels to 6h timeframe
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3_1d)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3_1d)
    camarilla_R4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R4_1d)
    camarilla_S4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S4_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 50)  # volume MA, weekly EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1w EMA50
        bullish_bias = curr_close > ema_1w_aligned[i]
        bearish_bias = curr_close < ema_1w_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above R3 with bullish bias AND volume spike
            # OR price breaks above R4 (strong breakout) regardless of bias
            long_entry = ((curr_high > camarilla_R3_aligned[i] and bullish_bias) or 
                         (curr_high > camarilla_R4_aligned[i])) and vol_spike
            
            # Short: price breaks below S3 with bearish bias AND volume spike
            # OR price breaks below S4 (strong breakdown) regardless of bias
            short_entry = ((curr_low < camarilla_S3_aligned[i] and bearish_bias) or 
                          (curr_low < camarilla_S4_aligned[i])) and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below S3 (mean reversion) OR loss of bullish bias
            if (curr_low < camarilla_S3_aligned[i]) or (curr_close < ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above R3 (mean reversion) OR loss of bearish bias
            if (curr_high > camarilla_R3_aligned[i]) or (curr_close > ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0