#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_rsi_volume_v1
Hypothesis: On 12h timeframe, use Camarilla pivot levels from 1-day timeframe for mean reversion entries.
Enter long when price touches S1/S2 support with RSI < 30 and volume > 1.5x average.
Enter short when price touches R1/R2 resistance with RSI > 70 and volume > 1.5x average.
Exit when price crosses the pivot point (mean reversion complete) or RSI returns to neutral (40-60).
Uses daily pivot levels for statistical significance in ranging markets, with volume confirmation to avoid false breaks.
Designed for low frequency (12-37 trades/year) to minimize fee drag in ranging/volatile markets like 2025.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_rsi_volume_v1"
timeframe = "12h"
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
    
    # RSI (14-period)
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get daily data for Camarilla pivot calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from daily OHLC
    # Camarilla formulas: 
    # H4 = Close + 1.5 * (High - Low)
    # H3 = Close + 1.0 * (High - Low)
    # H2 = Close + 0.5 * (High - Low)
    # H1 = Close + 0.25 * (High - Low)
    # L1 = Close - 0.25 * (High - Low)
    # L2 = Close - 0.5 * (High - Low)
    # L3 = Close - 1.0 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    pivot_range = daily_high - daily_low
    r4 = daily_close + 1.5 * pivot_range
    r3 = daily_close + 1.0 * pivot_range
    r2 = daily_close + 0.5 * pivot_range
    r1 = daily_close + 0.25 * pivot_range
    s1 = daily_close - 0.25 * pivot_range
    s2 = daily_close - 0.5 * pivot_range
    s3 = daily_close - 1.0 * pivot_range
    s4 = daily_close - 1.5 * pivot_range
    pivot = daily_close  # Camarilla uses close as pivot
    
    # Align Camarilla levels to 12h timeframe (shifted by 1 day to avoid look-ahead)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2)
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after RSI warmup
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0 or
            np.isnan(r1_12h[i]) or np.isnan(r2_12h[i]) or 
            np.isnan(s1_12h[i]) or np.isnan(s2_12h[i]) or
            np.isnan(pivot_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # RSI conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_neutral = (rsi[i] >= 40) & (rsi[i] <= 60)
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when price crosses above pivot (mean reversion complete)
            if close[i] > pivot_12h[i]:
                exit_long = True
            # Exit when RSI returns to neutral
            elif rsi_neutral[i]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit when price crosses below pivot (mean reversion complete)
            if close[i] < pivot_12h[i]:
                exit_short = True
            # Exit when RSI returns to neutral
            elif rsi_neutral[i]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price at S1/S2 support, RSI oversold, volume confirmation
            # Allow small tolerance for touching the level (0.1% of price)
            tolerance = 0.001 * close[i]
            long_entry = (
                ((abs(close[i] - s1_12h[i]) <= tolerance) or (abs(close[i] - s2_12h[i]) <= tolerance)) and
                rsi_oversold and vol_confirm
            )
            
            # Short entry: price at R1/R2 resistance, RSI overbought, volume confirmation
            short_entry = (
                ((abs(close[i] - r1_12h[i]) <= tolerance) or (abs(close[i] - r2_12h[i]) <= tolerance)) and
                rsi_overbought and vol_confirm
            )
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals