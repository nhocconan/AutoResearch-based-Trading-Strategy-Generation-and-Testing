#!/usr/bin/env python3
# 4h_12h_camarilla_volume_reversal_v1
# Strategy: 4-hour Camarilla pivot reversal with volume confirmation and 12-hour trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels (L3, L3, H3, H4) act as strong support/resistance.
# Long setup: Price closes below L3 with volume spike, then reverses back above L3 with 12h uptrend.
# Short setup: Price closes above H3 with volume spike, then reverses back below H3 with 12h downtrend.
# Uses tight reversal logic to limit trades (~20-40/year) and avoid fee drag.
# Works in ranging markets (mean reversion at extremes) and trending markets (pullbacks to pivot levels).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_volume_reversal_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for pivot calculation and trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for 12h timeframe
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Previous 12h bar's OHLC for Camarilla calculation
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    prev_close_12h = np.roll(close_12h, 1)
    # Set first value to NaN (no previous bar)
    prev_high_12h[0] = np.nan
    prev_low_12h[0] = np.nan
    prev_close_12h[0] = np.nan
    
    # Camarilla levels: based on previous day's range
    # H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low)
    # L3 = close - 1.1*(high-low), L4 = close - 1.5*(high-low)
    prev_range_12h = prev_high_12h - prev_low_12h
    camarilla_h4 = prev_close_12h + 1.5 * prev_range_12h
    camarilla_h3 = prev_close_12h + 1.1 * prev_range_12h
    camarilla_l3 = prev_close_12h - 1.1 * prev_range_12h
    camarilla_l4 = prev_close_12h - 1.5 * prev_range_12h
    
    # Align Camarilla levels to 4h timeframe (using previous 12h bar's close time)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    
    # 12h EMA(25) for trend filter (aligned to 4h)
    ema_25_12h = pd.Series(close_12h).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_25_12h)
    
    # 4h Volume confirmation: current volume > 2.0x 20-period average (tighter for fewer trades)
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (2.0 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(25, n):  # Start after EMA warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_25_12h_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Price reversal signals: close beyond level then back inside
        # Long: price was below L3, now closes back above L3
        long_reversal = (close[i-1] < camarilla_l3_aligned[i-1]) and (close[i] > camarilla_l3_aligned[i])
        # Short: price was above H3, now closes back below H3
        short_reversal = (close[i-1] > camarilla_h3_aligned[i-1]) and (close[i] < camarilla_h3_aligned[i])
        
        # Trend filter: price relative to 12h EMA25
        uptrend = close[i] > ema_25_12h_aligned[i]
        downtrend = close[i] < ema_25_12h_aligned[i]
        
        # Entry logic: reversal + volume + trend alignment
        if long_reversal and vol_confirm[i] and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_reversal and vol_confirm[i] and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite reversal signal with volume (mean reversion complete)
        elif position == 1 and short_reversal and vol_confirm[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and long_reversal and vol_confirm[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals