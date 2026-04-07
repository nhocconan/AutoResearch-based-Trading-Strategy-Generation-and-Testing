#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Camarilla Pivot + 1d Trend + Volume Spike
# Hypothesis: Camarilla pivot levels from daily data act as strong support/resistance.
# In uptrend (price > 1d EMA50), buy at L3 level with volume spike.
# In downtrend (price < 1d EMA50), sell at H3 level with volume spike.
# Uses 4h timeframe for precision, 1d for trend filter.
# Target: 20-50 trades per year (80-200 over 4 years).

name = "4h_camarilla_pivot_1d_trend_volume_v2"
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
    
    # Get 1d data for Camarilla pivot and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Formula: based on previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Avoid NaN from shift
    prev_high = np.where(np.isnan(prev_high), prev_close, prev_high)
    prev_low = np.where(np.isnan(prev_low), prev_close, prev_low)
    prev_close = np.where(np.isnan(prev_close), close[:len(prev_close)], prev_close)
    
    # Calculate pivot and levels
    pivot = (prev_high + prev_low + prev_close) / 3
    range_ = prev_high - prev_low
    
    # Camarilla levels
    L3 = pivot - (range_ * 1.1 / 4)
    L4 = pivot - (range_ * 1.1 / 2)
    H3 = pivot + (range_ * 1.1 / 4)
    H4 = pivot + (range_ * 1.1 / 2)
    
    # Align to 4h timeframe (already shifted by 1 in calculation)
    L3_4h = align_htf_to_ltf(prices, df_1d, L3)
    L4_4h = align_htf_to_ltf(prices, df_1d, L4)
    H3_4h = align_htf_to_ltf(prices, df_1d, H3)
    H4_4h = align_htf_to_ltf(prices, df_1d, H4)
    
    # 1d trend filter: EMA50
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(L3_4h[i]) or np.isnan(H3_4h[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below L3 OR trend turns bearish
            if close[i] < L3_4h[i] or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses above H3 OR trend turns bullish
            if close[i] > H3_4h[i] or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price crosses above L3 with bullish trend
                if close[i] > L3_4h[i] and (i == 50 or close[i-1] <= L3_4h[i-1]) and close[i] > ema_50_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price crosses below H3 with bearish trend
                elif close[i] < H3_4h[i] and (i == 50 or close[i-1] >= H3_4h[i-1]) and close[i] < ema_50_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals