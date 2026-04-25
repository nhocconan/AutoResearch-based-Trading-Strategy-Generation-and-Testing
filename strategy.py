#!/usr/bin/env python3
"""
4h Camarilla R3 S3 Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Camarilla pivot levels act as strong support/resistance. Breakouts above R3 or below S3 with volume confirmation and 1d EMA trend filter capture institutional moves. Works in bull markets (breakouts with trend) and bear markets (avoids false breakouts via trend filter). Target: 25-40 trades/year on 4h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar (for intraday use)
    # We'll use the 1d OHLC from the previous completed day
    camarilla_h3 = np.zeros(n)
    camarilla_l3 = np.zeros(n)
    camarilla_h4 = np.zeros(n)
    camarilla_l4 = np.zeros(n)
    
    # For each 4h bar, we need the previous 1d bar's OHLC
    # Since we're on 4h timeframe, we can use the 1d data aligned
    # Camarilla levels: H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    #                   H3 = close + 1.125*(high-low), L3 = close - 1.125*(high-low)
    # But we need to use the previous day's values, not current day's
    # So we shift the 1d data by 1 bar
    if len(df_1d) >= 2:
        prev_close_1d = np.roll(df_1d['close'].values, 1)
        prev_high_1d = np.roll(df_1d['high'].values, 1)
        prev_low_1d = np.roll(df_1d['low'].values, 1)
        prev_close_1d[0] = df_1d['close'].values[0]  # first value
        prev_high_1d[0] = df_1d['high'].values[0]
        prev_low_1d[0] = df_1d['low'].values[0]
        
        camarilla_h3_1d = prev_close_1d + 1.125 * (prev_high_1d - prev_low_1d)
        camarilla_l3_1d = prev_close_1d - 1.125 * (prev_high_1d - prev_low_1d)
        camarilla_h4_1d = prev_close_1d + 1.5 * (prev_high_1d - prev_low_1d)
        camarilla_l4_1d = prev_close_1d - 1.5 * (prev_high_1d - prev_low_1d)
        
        # Align to 4h timeframe
        camarilla_h3 = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
        camarilla_l3 = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
        camarilla_h4 = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d)
        camarilla_l4 = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for volume MA and Camarilla alignment
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        camarilla_h3_val = camarilla_h3[i]
        camarilla_l3_val = camarilla_l3[i]
        camarilla_h4_val = camarilla_h4[i]
        camarilla_l4_val = camarilla_l4[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Camarilla H3 AND volume spike AND price > 1d EMA34 (uptrend)
            long_entry = (curr_close > camarilla_h3_val) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below Camarilla L3 AND volume spike AND price < 1d EMA34 (downtrend)
            short_entry = (curr_close < camarilla_l3_val) and vol_spike and (curr_close < ema_trend)
            
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
            # Exit: price crosses below Camarilla L3 OR price < 1d EMA34 (trend change)
            if (curr_close < camarilla_l3_val) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above Camarilla H3 OR price > 1d EMA34 (trend change)
            if (curr_close > camarilla_h3_val) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0