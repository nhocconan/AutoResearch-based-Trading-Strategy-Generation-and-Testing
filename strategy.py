#!/usr/bin/env python3
"""
6h_camarilla_pivot_1d_volume_v1
Hypothesis: On 6-hour timeframe, trade reversals at daily Camarilla pivot levels (R3/S3) with volume confirmation.
Go long when price closes above S3 with volume > 1.5x 20-period average and RSI < 40.
Go short when price closes below R3 with volume > 1.5x 20-period average and RSI > 60.
Exit when price crosses the daily pivot point (PP).
Designed for 15-30 trades/year to minimize fee decay while capturing mean reversions at key institutional levels.
Works in both bull/bear markets as Camarilla levels adapt to volatility and RSI filter avoids overextended moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_volume_v1"
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
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for today's pivots
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # first day uses same day
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_ = prev_high - prev_low
    
    # Camarilla levels
    S3 = pivot - (1.1 * range_ / 2)
    S2 = pivot - (1.1 * range_ / 4)
    S1 = pivot - (1.1 * range_ / 6)
    PP = pivot
    R1 = pivot + (1.1 * range_ / 6)
    R2 = pivot + (1.1 * range_ / 4)
    R3 = pivot + (1.1 * range_ / 2)
    
    # Align to 6h timeframe
    S3_6h = align_htf_to_ltf(prices, df_1d, S3)
    R3_6h = align_htf_to_ltf(prices, df_1d, R3)
    PP_6h = align_htf_to_ltf(prices, df_1d, PP)
    
    # Calculate RSI on 6h timeframe
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(20, 14), n):
        # Skip if data not available
        if (np.isnan(S3_6h[i]) or np.isnan(R3_6h[i]) or np.isnan(PP_6h[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below daily pivot
            if close[i] < PP_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above daily pivot
            if close[i] > PP_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation
            if vol_ok:
                # Long: price closes above S3 with RSI < 40 (oversold)
                if (close[i] > S3_6h[i] and close[i-1] <= S3_6h[i-1] and 
                    rsi[i] < 40):
                    position = 1
                    signals[i] = 0.25
                # Short: price closes below R3 with RSI > 60 (overbought)
                elif (close[i] < R3_6h[i] and close[i-1] >= R3_6h[i-1] and 
                      rsi[i] > 60):
                    position = -1
                    signals[i] = -0.25
    
    return signals