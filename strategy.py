#!/usr/bin/env python3
"""
12h Camarilla R1/S1 Breakout with 1w EMA34 Trend and Volume Spike
Hypothesis: Camarilla pivot levels (R1/S1) act as intraday support/resistance on 12h chart.
Breakout above R1 with 1w uptrend (EMA34) and volume spike signals bullish momentum.
Breakdown below S1 with 1w downtrend and volume spike signals bearish momentum.
Uses 12h primary timeframe with 1w HTF for trend filter. Targets 50-150 total trades over 4 years.
Works in both bull (breakouts with trend) and bear (breakdowns with trend) markets.
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
    
    # Get 1w data for EMA34 trend (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1w close for trend
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d OHLC for Camarilla pivot levels (R1, S1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Camarilla levels: based on previous day's OHLC
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    camarilla_R1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    camarilla_S1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # Align Camarilla levels to 12h timeframe (extra delay for daily pivot confirmation)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1, additional_delay_bars=1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1, additional_delay_bars=1)
    
    # Calculate 20-period volume MA for 12h volume confirmation
    vol_ma_20_12h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_12h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA, Camarilla, and volume MA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or
            np.isnan(vol_ma_20_12h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1w_aligned[i]
        R1 = camarilla_R1_aligned[i]
        S1 = camarilla_S1_aligned[i]
        vol_ma_12h = vol_ma_20_12h[i]
        
        # Volume confirmation: current 12h volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma_12h
        
        if position == 0:
            # Look for entry signals
            # Long: Break above R1 AND price > EMA34 (uptrend) AND volume confirmation
            long_entry = (curr_high > R1 and 
                         curr_close > ema_trend and volume_confirm)
            # Short: Break below S1 AND price < EMA34 (downtrend) AND volume confirmation
            short_entry = (curr_low < S1 and 
                          curr_close < ema_trend and volume_confirm)
            
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
            # Exit: Price crosses below S1 (reversal) OR price falls below EMA34
            if (curr_close < S1 or curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Price crosses above R1 (reversal) OR price rises above EMA34
            if (curr_close > R1 or curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0