#!/usr/bin/env python3
# 4h_camarilla_pivot_1d_trend_volume_v1
# Hypothesis: On 4h timeframe, use daily Camarilla pivot levels with trend filter and volume confirmation.
# Long when price closes above H3 (bullish pivot) with volume > 1.5x average and daily trend up.
# Short when price closes below L3 (bearish pivot) with volume > 1.5x average and daily trend down.
# Exit on opposite pivot touch or when volume drops below average.
# Daily trend defined by price above/below daily EMA20.
# This strategy targets fewer trades (19-50/year) by using higher timeframe structure and tight entry conditions.
# Works in both bull and bear markets via trend filter and pivot mean reversion in ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_1d_trend_volume_v1"
timeframe = "4h"
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
    
    # Calculate pivot points from daily data (using previous day's OHLC)
    # Get daily data
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels using previous day's data
    # Camarilla formulas: H4 = C + (H-L)*1.1/2, H3 = C + (H-L)*1.1/4, etc.
    # We use previous day's H, L, C to avoid look-ahead
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Calculate pivot levels using previous day's data
    camarilla_H4 = daily_close + (daily_high - daily_low) * 1.1 / 2
    camarilla_H3 = daily_close + (daily_high - daily_low) * 1.1 / 4
    camarilla_H2 = daily_close + (daily_high - daily_low) * 1.1 / 6
    camarilla_H1 = daily_close + (daily_high - daily_low) * 1.1 / 12
    camarilla_L1 = daily_close - (daily_high - daily_low) * 1.1 / 12
    camarilla_L2 = daily_close - (daily_high - daily_low) * 1.1 / 6
    camarilla_L3 = daily_close - (daily_high - daily_low) * 1.1 / 4
    camarilla_L4 = daily_close - (daily_high - daily_low) * 1.1 / 2
    
    # Align daily pivot levels to 4h timeframe (with proper delay for daily bar close)
    H4_4h = align_htf_to_ltf(prices, df_daily, camarilla_H4)
    H3_4h = align_htf_to_ltf(prices, df_daily, camarilla_H3)
    L3_4h = align_htf_to_ltf(prices, df_daily, camarilla_L3)
    
    # Daily trend filter: price above/below daily EMA20
    daily_ema20 = pd.Series(daily_close).ewm(span=20, min_periods=20, adjust=False).mean().values
    daily_ema20_4h = align_htf_to_ltf(prices, df_daily, daily_ema20)
    
    # Volume confirmation: 20-period average on 4h
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(H3_4h[i]) or np.isnan(L3_4h[i]) or np.isnan(daily_ema20_4h[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price touches L3 (opposite pivot) or volume drops below average
            if close[i] <= L3_4h[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches H3 (opposite pivot) or volume drops below average
            if close[i] >= H3_4h[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Daily trend filter
            daily_uptrend = close[i] > daily_ema20_4h[i]
            daily_downtrend = close[i] < daily_ema20_4h[i]
            
            # Long entry: price closes above H3 with volume and uptrend
            if close[i] > H3_4h[i] and volume_ok and daily_uptrend:
                position = 1
                signals[i] = 0.25
            # Short entry: price closes below L3 with volume and downtrend
            elif close[i] < L3_4h[i] and volume_ok and daily_downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals