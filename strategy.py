#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_Reversal_v1
Hypothesis: Trade reversals at daily Camarilla H3/L3 levels with volume confirmation and RSI filter.
Long when price rejects L3 with volume surge and RSI<30 (oversold).
Short when price rejects H3 with volume surge and RSI>70 (overbought).
Exit on opposite level touch or RSI normalization (40-60 range).
Designed for low frequency (<25 trades/year) with high conviction in mean reversion.
Works in ranging markets (reversions) and trending markets (pullbacks to pivot).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Pivot_Reversal_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA FOR CAMARILLA PIVOTS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    typical_price = (high_1d + low_1d + close_1d) / 3
    pivot = typical_price
    range_1d = high_1d - low_1d
    
    h3 = pivot + (range_1d * 1.1 / 4)
    l3 = pivot - (range_1d * 1.1 / 4)
    h4 = pivot + (range_1d * 1.1 / 2)
    l4 = pivot - (range_1d * 1.1 / 2)
    
    # Align to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # === RSI FILTER (14-period) ===
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # === VOLUME FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume strength (must be significantly above average)
        strong_volume = volume[i] > (vol_ma[i] * 1.5)
        
        # Rejection conditions: price touches level but closes back inside
        # Long: price touches/goes below L3 but closes above it (bounce)
        touch_l3 = low[i] <= l3_aligned[i]
        close_above_l3 = close[i] > l3_aligned[i]
        long_setup = touch_l3 and close_above_l3
        
        # Short: price touches/goes above H3 but closes below it (rejection)
        touch_h3 = high[i] >= h3_aligned[i]
        close_below_h3 = close[i] < h3_aligned[i]
        short_setup = touch_h3 and close_below_h3
        
        # RSI extremes for mean reversion
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_neutral = (rsi[i] >= 40) & (rsi[i] <= 60)
        
        # Entry signals
        long_signal = long_setup and strong_volume and rsi_oversold
        short_signal = short_setup and strong_volume and rsi_overbought
        
        # Exit conditions
        exit_long = (position == 1 and 
                    (close[i] >= h3_aligned[i] or rsi_neutral[i]))
        exit_short = (position == -1 and 
                     (close[i] <= l3_aligned[i] or rsi_neutral[i]))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals