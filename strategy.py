#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_atr_volume_v1
Hypothesis: Camarilla pivot levels from daily chart provide key support/resistance.
Price often reverses or breaks from these levels with volume confirmation.
In trending markets, breakouts through S3/R3 with volume continue the trend.
In ranging markets, reversals from S1/R1, S2/R2 with volume offer mean reversion.
Trades both breakouts and reversals based on price action at Camarilla levels.
Uses ATR for volatility filtering and position sizing.
Target: 20-40 trades/year (80-160 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_atr_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    H4 = prev_close + 1.5 * (prev_high - prev_low)
    H3 = prev_close + 1.0 * (prev_high - prev_low)
    H2 = prev_close + 0.5 * (prev_high - prev_low)
    H1 = prev_close + 0.25 * (prev_high - prev_low)
    L1 = prev_close - 0.25 * (prev_high - prev_low)
    L2 = prev_close - 0.5 * (prev_high - prev_low)
    L3 = prev_close - 1.0 * (prev_high - prev_low)
    L4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Align all levels to 12h timeframe (shifted by 1 day for lookback)
    H4_12h = align_htf_to_ltf(prices, df_1d, H4)
    H3_12h = align_htf_to_ltf(prices, df_1d, H3)
    H2_12h = align_htf_to_ltf(prices, df_1d, H2)
    H1_12h = align_htf_to_ltf(prices, df_1d, H1)
    L1_12h = align_htf_to_ltf(prices, df_1d, L1)
    L2_12h = align_htf_to_ltf(prices, df_1d, L2)
    L3_12h = align_htf_to_ltf(prices, df_1d, L3)
    L4_12h = align_htf_to_ltf(prices, df_1d, L4)
    
    # ATR for volatility filter and position sizing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_series = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    atr = atr_series.values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Volatility filter: ATR > 0.5 * 50-period ATR average (avoid choppy low-vol periods)
    atr_ma_series = pd.Series(atr).rolling(window=50, min_periods=50).mean()
    atr_ma = atr_ma_series.values
    vol_filter = atr > (0.5 * atr_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any data is not ready
        if (np.isnan(H3_12h[i]) or np.isnan(L3_12h[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or np.isnan(atr_ma[i])):
            signals[i] = 0.0
            continue
        
        # Apply volatility and volume filters
        if not (volume_spike[i] and vol_filter[i]):
            if position != 0:
                # Exit if filters fail
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below L2 (strong support broken)
            if close[i] < L2_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above H2 (strong resistance broken)
            if close[i] > H2_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above H3 with volume (bullish breakout)
            # OR price bounces from L3/L4 with volume (bullish reversal)
            long_breakout = (close[i] > H3_12h[i] and close[i-1] <= H3_12h[i-1])
            long_bounce = ((close[i] > L3_12h[i] and close[i-1] <= L3_12h[i-1]) or 
                          (close[i] > L4_12h[i] and close[i-1] <= L4_12h[i-1]))
            if (long_breakout or long_bounce) and close[i] < H2_12h[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below L3 with volume (bearish breakout)
            # OR price rejects from H3/H4 with volume (bearish reversal)
            elif ((close[i] < L3_12h[i] and close[i-1] >= L3_12h[i-1]) or  # Breakdown below L3
                  ((close[i] < H3_12h[i] and close[i-1] >= H3_12h[i-1]) or  # Rejection from H3
                   (close[i] < H4_12h[i] and close[i-1] >= H4_12h[i-1])) and  # Rejection from H4
                  close[i] > L2_12h[i]):  # But not below strong support
                position = -1
                signals[i] = -0.25
    
    return signals