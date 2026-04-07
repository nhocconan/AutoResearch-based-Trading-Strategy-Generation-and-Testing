#!/usr/bin/env python3
"""
6h_camarilla_pivot_1d_rsi_volume_v1
Hypothesis: On 6-hour timeframe, use Camarilla pivot levels from 1-day timeframe to identify key support/resistance. 
Enter long at S3/S4 with RSI < 30 and volume confirmation; enter short at R3/R4 with RSI > 70 and volume confirmation.
Exit on opposite RSI crossover (RSI > 50 for longs, RSI < 50 for shorts). 
Camarilla levels provide institutional reference points, RSI filters avoid false breakouts, and volume confirms participation.
Designed for 50-150 total trades over 4 years (~12-37/year) to minimize fee drag while performing in both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_rsi_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if df_1d is None or len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 1-day bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: H4/L4 = close ± range * 1.1/2, H3/L3 = close ± range * 1.1/4, etc.
    # We use H3, L3, H4, L4 for entries
    h3 = close_1d + range_1d * 1.1 / 4
    l3 = close_1d - range_1d * 1.1 / 4
    h4 = close_1d + range_1d * 1.1 / 2
    l4 = close_1d - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (shifted by 1 day for completed bars only)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Calculate RSI on 6h timeframe
    rsi_period = 14
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume filter: 20-period average on 6h timeframe
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(rsi_period, 20), n):
        # Skip if data not available
        if (np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation: require volume above average
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: RSI crosses above 50 (overbought)
            if rsi[i] > 50 and rsi[i-1] <= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI crosses below 50 (oversold)
            if rsi[i] < 50 and rsi[i-1] >= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long entry: price at or below S3/L3 with RSI oversold (<30)
                if close[i] <= l3_aligned[i] and rsi[i] < 30:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price at or above R3/H3 with RSI overbought (>70)
                elif close[i] >= h3_aligned[i] and rsi[i] > 70:
                    position = -1
                    signals[i] = -0.25
    
    return signals