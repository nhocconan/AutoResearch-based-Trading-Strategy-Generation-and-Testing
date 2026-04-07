#!/usr/bin/env python3
"""
12h_camarilla_pivot_1w_trend_volume_v1
Hypothesis: On 12h timeframe, use weekly Camarilla pivot levels with volume confirmation and trend filter. 
Enter long when price breaks above H3 with volume > 1.5x average and price above 12h EMA50; enter short when price breaks below L3 with volume > 1.5x average and price below 12h EMA50. 
Exit on opposite signal or when price returns to Pivot level. Weekly pivots provide stronger support/resistance, reducing trades and increasing win rate. Targets 12-37 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA50 for trend filter
    ema50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for Camarilla pivots (calculate once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each week: based on previous week's OHLC
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    # Calculate pivot levels
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels
    H3 = pivot + 1.0 * range_val
    L3 = pivot - 1.0 * range_val
    H4 = pivot + 1.5 * range_val
    L4 = pivot - 1.5 * range_val
    
    # Align to 12h timeframe (shifted by 1 week to avoid look-ahead)
    H3_aligned = align_htf_to_ltf(prices, df_1w, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1w, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1w, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1w, L4)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema50[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0 or
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit on short signal (price breaks below L3 with volume and trend)
            if close[i] < L3_aligned[i] and vol_confirm and close[i] < ema50[i]:
                exit_long = True
            # Exit when price returns to pivot level (mean reversion)
            elif abs(close[i] - pivot_aligned[i]) < 0.5 * (high[i] - low[i]):
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit on long signal (price breaks above H3 with volume and trend)
            if close[i] > H3_aligned[i] and vol_confirm and close[i] > ema50[i]:
                exit_short = True
            # Exit when price returns to pivot level (mean reversion)
            elif abs(close[i] - pivot_aligned[i]) < 0.5 * (high[i] - low[i]):
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above H3 with volume confirmation and uptrend
            long_entry = close[i] > H3_aligned[i] and vol_confirm and close[i] > ema50[i]
            
            # Short entry: price breaks below L3 with volume confirmation and downtrend
            short_entry = close[i] < L3_aligned[i] and vol_confirm and close[i] < ema50[i]
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals