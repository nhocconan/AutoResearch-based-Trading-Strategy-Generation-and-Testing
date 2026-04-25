#!/usr/bin/env python3
"""
12h Camarilla R1/S1 Breakout with 1d EMA34 Trend and Volume Spike
Hypothesis: Camarilla pivot levels (R1/S1) act as strong intraday support/resistance. 
A break above R1 with 1d uptrend and volume spike signals bullish momentum; 
break below S1 with 1d downtrend and volume spike signals bearish momentum.
Uses 12h timeframe with 1d HTF for trend filter. Targets 50-150 total trades over 4 years (12-37/year).
Works in bull markets via breakout continuation and in bear markets via mean reversion at extremes.
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
    
    # Get 1d data for EMA34 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d close for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla pivot levels (based on previous 1d bar)
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_R1_1d = df_1d['close'].values + (1.1/12) * (df_1d['high'].values - df_1d['low'].values)
    camarilla_S1_1d = df_1d['close'].values - (1.1/12) * (df_1d['high'].values - df_1d['low'].values)
    
    # Align Camarilla levels to 12h timeframe (no extra delay - levels known after 1d close)
    camarilla_R1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1_1d)
    camarilla_S1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1_1d)
    
    # Calculate 24-period volume MA for 12h volume confirmation (24 periods = 12 days of 12h bars)
    vol_ma_24_12h = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma_24_12h[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA34 and volume MA
    start_idx = max(34, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_R1_1d_aligned[i]) or np.isnan(camarilla_S1_1d_aligned[i]) or
            np.isnan(vol_ma_24_12h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        camarilla_R1 = camarilla_R1_1d_aligned[i]
        camarilla_S1 = camarilla_S1_1d_aligned[i]
        vol_ma_12h = vol_ma_24_12h[i]
        
        # Volume confirmation: current 12h volume > 2.0 * 24-period average
        volume_confirm = curr_volume > 2.0 * vol_ma_12h
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above R1 AND 1d uptrend AND volume confirmation
            long_entry = (curr_high > camarilla_R1 and 
                         ema_trend > 0 and volume_confirm)
            # Short: price breaks below S1 AND 1d downtrend AND volume confirmation
            short_entry = (curr_low < camarilla_S1 and 
                          ema_trend < 0 and volume_confirm)
            
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
            # Exit: price falls back below R1 OR 1d trend turns down
            if (curr_close < camarilla_R1 or ema_trend < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises back above S1 OR 1d trend turns up
            if (curr_close > camarilla_S1 or ema_trend > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0