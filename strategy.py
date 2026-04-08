#!/usr/bin/env python3
# 6h_camarilla_pivot_1d_ema_volume_v1
# Hypothesis: 6-hour Camarilla pivot levels from 1-day timeframe with EMA trend filter and volume confirmation.
# Long when price breaks above H4 level with volume > 1.5x average and price > EMA50 (bullish trend).
# Short when price breaks below L4 level with volume > 1.5x average and price < EMA50 (bearish trend).
# Exit when price returns to H3/L3 levels or opposite Camarilla level is broken.
# Designed to work in both bull and bear markets by using intraday pivot levels that adapt to volatility.
# Target: 15-25 trades/year to minimize fee decay while capturing institutional levels.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA50 for trend filter
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1-day data for Camarilla pivots (calculate once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivots from previous day's OHLC
    # Camarilla formula: 
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # H2 = close + 0.6 * (high - low)
    # H1 = close + 0.318 * (high - low)
    # L1 = close - 0.318 * (high - low)
    # L2 = close - 0.6 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # L4 = close - 1.5 * (high - low)
    
    prev_close = df_1d['close'].shift(1).values  # Previous day's close
    prev_high = df_1d['high'].shift(1).values    # Previous day's high
    prev_low = df_1d['low'].shift(1).values      # Previous day's low
    
    # Calculate pivot levels
    H4 = prev_close + 1.5 * (prev_high - prev_low)
    H3 = prev_close + 1.1 * (prev_high - prev_low)
    L3 = prev_close - 1.1 * (prev_high - prev_low)
    L4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Align pivots to 6h timeframe (wait for previous day to complete)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Volume confirmation: 50-period average
    avg_volume = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(H4_aligned[i]) or np.isnan(H3_aligned[i]) or \
           np.isnan(L3_aligned[i]) or np.isnan(L4_aligned[i]) or \
           np.isnan(ema50[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to H3 level or breaks below L4 (strong reversal)
            if close[i] <= H3_aligned[i] or close[i] < L4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to L3 level or breaks above H4 (strong reversal)
            if close[i] >= L3_aligned[i] or close[i] > H4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Long entry: price breaks above H4 with volume and trend filter (price > EMA50)
            if close[i] > H4_aligned[i] and volume_ok and close[i] > ema50[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below L4 with volume and trend filter (price < EMA50)
            elif close[i] < L4_aligned[i] and volume_ok and close[i] < ema50[i]:
                position = -1
                signals[i] = -0.25
    
    return signals