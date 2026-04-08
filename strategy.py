#!/usr/bin/env python3
# 1d_weekly_camarilla_trend_volume_v1
# Hypothesis: On daily timeframe, use weekly Camarilla pivot levels with trend filter and volume confirmation.
# Long when price closes above H3 (bullish pivot) with volume > 1.5x average and weekly trend up.
# Short when price closes below L3 (bearish pivot) with volume > 1.5x average and weekly trend down.
# Exit on opposite pivot touch or when volume drops below average.
# Weekly trend defined by price above/below weekly EMA20.
# Target: 30-100 trades over 4 years (7-25/year) to minimize fee drag and work in both bull/bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_camarilla_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data (HTF)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels using previous week's OHLC (no look-ahead)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Camarilla formulas based on previous week's range
    camarilla_H4 = weekly_close + (weekly_high - weekly_low) * 1.1 / 2
    camarilla_H3 = weekly_close + (weekly_high - weekly_low) * 1.1 / 4
    camarilla_H2 = weekly_close + (weekly_high - weekly_low) * 1.1 / 6
    camarilla_H1 = weekly_close + (weekly_high - weekly_low) * 1.1 / 12
    camarilla_L1 = weekly_close - (weekly_high - weekly_low) * 1.1 / 12
    camarilla_L2 = weekly_close - (weekly_high - weekly_low) * 1.1 / 6
    camarilla_L3 = weekly_close - (weekly_high - weekly_low) * 1.1 / 4
    camarilla_L4 = weekly_close - (weekly_high - weekly_low) * 1.1 / 2
    
    # Align weekly pivot levels to daily timeframe (properly delayed for weekly bar close)
    H3_1d = align_htf_to_ltf(prices, df_weekly, camarilla_H3)
    L3_1d = align_htf_to_ltf(prices, df_weekly, camarilla_L3)
    
    # Weekly trend filter: price above/below weekly EMA20
    weekly_ema20 = pd.Series(weekly_close).ewm(span=20, min_periods=20, adjust=False).mean().values
    weekly_ema20_1d = align_htf_to_ltf(prices, df_weekly, weekly_ema20)
    
    # Volume confirmation: 20-day average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(H3_1d[i]) or np.isnan(L3_1d[i]) or np.isnan(weekly_ema20_1d[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price touches L3 (opposite pivot) or volume drops below average
            if close[i] <= L3_1d[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches H3 (opposite pivot) or volume drops below average
            if close[i] >= H3_1d[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Weekly trend filter
            weekly_uptrend = close[i] > weekly_ema20_1d[i]
            weekly_downtrend = close[i] < weekly_ema20_1d[i]
            
            # Long entry: price closes above H3 with volume and uptrend
            if close[i] > H3_1d[i] and volume_ok and weekly_uptrend:
                position = 1
                signals[i] = 0.25
            # Short entry: price closes below L3 with volume and downtrend
            elif close[i] < L3_1d[i] and volume_ok and weekly_downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals