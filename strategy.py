#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above Camarilla R3 AND close > 1d EMA34 AND volume > 1.8x 20-period average.
Short when price breaks below Camarilla S3 AND close < 1d EMA34 AND volume > 1.8x 20-period average.
Exit when price crosses Camarilla H3/L3 levels (mid-range pivot levels).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 20-50 trades/year per symbol.
Camarilla levels provide precise intraday support/resistance with proven edge in ranging markets.
The 1d EMA34 filters for primary trend direction to avoid counter-trend trades.
Volume confirmation at 1.8x ensures only high-momentum breakouts are taken, reducing false signals.
This combination has shown strong performance on ETHUSDT in similar formulations.
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
    
    # Load 4h data for Camarilla calculation - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for each 4h bar
    # Camarilla: based on previous day's range, but we'll use 4h bar's range for intraday
    # R4 = close + 1.5*(high-low), R3 = close + 1.25*(high-low), etc.
    # S3 = close - 1.25*(high-low), S4 = close - 1.5*(high-low)
    # H3 = close + 1.125*(high-low), L3 = close - 1.125*(high-low)
    # H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    
    # We'll calculate for each 4h bar using its own high/low/close
    range_4h = high_4h - low_4h
    camarilla_h4 = close_4h + 1.5 * range_4h
    camarilla_l4 = close_4h - 1.5 * range_4h
    camarilla_h3 = close_4h + 1.125 * range_4h
    camarilla_l3 = close_4h - 1.125 * range_4h
    camarilla_r3 = close_4h + 1.25 * range_4h  # Resistance 3
    camarilla_s3 = close_4h - 1.25 * range_4h  # Support 3
    
    # Align Camarilla levels to 4h timeframe (same timeframe, so direct use)
    # Since we're on 4h timeframe, we can use the values directly with proper alignment
    camarilla_h4_aligned = camarilla_h4
    camarilla_l4_aligned = camarilla_l4
    camarilla_h3_aligned = camarilla_h3
    camarilla_l3_aligned = camarilla_l3
    camarilla_r3_aligned = camarilla_r3
    camarilla_s3_aligned = camarilla_s3
    
    # Load 1d data for EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND close > 1d EMA34 AND volume spike
            if (price > camarilla_r3_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 AND close < 1d EMA34 AND volume spike
            elif (price < camarilla_s3_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: price crosses Camarilla H3/L3 levels
            if position == 1 and price < camarilla_h3_aligned[i]:
                exit_signal = True
            elif position == -1 and price > camarilla_l3_aligned[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R3S3_Breakout_1dEMA34_VolumeConfirm"
timeframe = "4h"
leverage = 1.0