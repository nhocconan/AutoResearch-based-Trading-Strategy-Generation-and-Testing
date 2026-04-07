#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Camarilla Pivot Reversal + Daily Trend + Volume Spike
# Hypothesis: Camarilla pivot levels (L3, L4, H3, H4) act as strong support/resistance.
# In ranging markets, price reverses at these levels with volume confirmation.
# Daily trend filter (EMA50) ensures trades align with higher-timeframe momentum.
# Designed for 12h timeframe with low trade frequency (12-37/year).
# Works in bull via long at L3/L4 in uptrend, short at H3/H4 in downtrend.
# Works in bear via short at H3/H4 in downtrend, long at L3/L4 in uptrend retracements.

name = "12h_camarilla_pivot_1d_trend_volume_v1"
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
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Formula: 
    # H4 = close + 1.5*(high - low)
    # H3 = close + 1.1*(high - low)
    # L3 = close - 1.1*(high - low)
    # L4 = close - 1.5*(high - low)
    # Using previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate pivot levels
    hl_range = prev_high - prev_low
    H4 = prev_close + 1.5 * hl_range
    H3 = prev_close + 1.1 * hl_range
    L3 = prev_close - 1.1 * hl_range
    L4 = prev_close - 1.5 * hl_range
    
    # Align pivot levels to 12h timeframe (shifted by 1 day to avoid look-ahead)
    H4_12h = align_htf_to_ltf(prices, df_1d, H4)
    H3_12h = align_htf_to_ltf(prices, df_1d, H3)
    L3_12h = align_htf_to_ltf(prices, df_1d, L3)
    L4_12h = align_htf_to_ltf(prices, df_1d, L4)
    
    # Daily trend filter: EMA(50) of daily close
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False).mean().values
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=15).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(H4_12h[i]) or np.isnan(L4_12h[i]) or 
            np.isnan(ema_50_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below L3 OR daily trend turns bearish
            if close[i] < L3_12h[i] or close[i] < ema_50_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses above H3 OR daily trend turns bullish
            if close[i] > H3_12h[i] or close[i] > ema_50_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price touches L4 with daily uptrend (bullish reversal)
                if (close[i] <= L4_12h[i] * 1.002) and (i == 50 or close[i-1] > L4_12h[i-1] * 1.002) and close[i] > ema_50_12h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price touches H4 with daily downtrend (bearish reversal)
                elif (close[i] >= H4_12h[i] * 0.998) and (i == 50 or close[i-1] < H4_12h[i-1] * 0.998) and close[i] < ema_50_12h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals