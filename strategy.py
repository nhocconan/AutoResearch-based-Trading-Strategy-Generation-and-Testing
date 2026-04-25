#!/usr/bin/env python3
"""
12h Camarilla H3L3 Breakout with 1d EMA50 Trend and Volume Spike
Hypothesis: Camarilla H3/L3 levels act as strong intraday support/resistance on 12h chart.
Breakouts above H3 or below L3 with volume confirmation and aligned 1d EMA50 trend capture
continuation moves. Works in both bull/bear markets: EMA50 filter ensures we trade with
higher timeframe momentum, while Camarilla breakouts provide precise entry points.
Volume spike confirms institutional participation. Designed for low trade frequency
(12-37/year) to minimize fee drag on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 1d close for trend
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period ATR for volume spike threshold
    tr1 = pd.Series(high).rolling(2).max() - pd.Series(low).rolling(2).min()
    tr2 = abs(pd.Series(high) - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low) - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 12h Camarilla levels (based on previous day's OHLC)
    # Need to resample to daily OHLC from 12h data
    # Since we're on 12h timeframe, we can use 2-bar lookback for daily OHLC
    # But better to get actual daily data
    df_1d_ohlc = get_htf_data(prices, '1d')
    if len(df_1d_ohlc) < 2:
        return np.zeros(n)
    
    # Get daily OHLC values
    d_high = df_1d_ohlc['high'].values
    d_low = df_1d_ohlc['low'].values
    d_close = df_1d_ohlc['close'].values
    
    # Calculate Camarilla levels for each day
    # H3 = close + (high - low) * 1.1/4
    # L3 = close - (high - low) * 1.1/4
    camarilla_h3 = d_close + (d_high - d_low) * 1.1 / 4
    camarilla_l3 = d_close - (d_high - d_low) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (each daily level lasts for 2 bars)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_l3)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA, ATR, volume MA, and Camarilla
    start_idx = max(50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_50_1d_aligned[i]
        h3_level = camarilla_h3_aligned[i]
        l3_level = camarilla_l3_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H3 level AND volume spike AND price > 1d EMA50 (uptrend)
            long_entry = (curr_close > h3_level) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below L3 level AND volume spike AND price < 1d EMA50 (downtrend)
            short_entry = (curr_close < l3_level) and vol_spike and (curr_close < ema_trend)
            
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
            # Exit: price crosses below L3 level (reversal) OR price crosses below EMA (trend change)
            if (curr_close < l3_level) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above H3 level (reversal) OR price crosses above EMA (trend change)
            if (curr_close > h3_level) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0