#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Camarilla Pivot Reversal with Volume and 1D Trend Filter
# Hypothesis: Price reversing from Camarilla pivot levels (H3/L3) on 12h timeframe
# indicates mean reversion. Volume confirms institutional participation at these levels.
# Trend filter (price above/below 100 EMA on 1D) ensures alignment with higher timeframe trend.
# Works in both bull and bear markets: in bull, only long reversals at L3; in bear, only short reversals at H3.
# Target: 12-37 trades/year (50-150 over 4 years).

name = "12h_camarilla_pivot_reversal_volume_trend_v1"
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
    
    # Get daily data for Camarilla pivot calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate daily data (previous day's OHLC for pivot)
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    prev_daily_high = np.roll(daily_high, 1)
    prev_daily_low = np.roll(daily_low, 1)
    prev_daily_close = np.roll(daily_close, 1)
    
    # Handle first element
    if len(prev_daily_high) > 1:
        prev_daily_high[0] = prev_daily_high[1]
        prev_daily_low[0] = prev_daily_low[1]
        prev_daily_close[0] = prev_daily_close[1]
    else:
        prev_daily_high[0] = 0
        prev_daily_low[0] = 0
        prev_daily_close[0] = 0
    
    # Calculate Camarilla pivot levels for previous day
    # Camarilla formulas:
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # We'll use H3 and L3 for reversal signals
    range_prev = prev_daily_high - prev_daily_low
    camarilla_h3 = prev_daily_close + 1.1 * range_prev
    camarilla_l3 = prev_daily_close - 1.1 * range_prev
    
    # Align to 12h timeframe (use previous day's levels)
    h3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_l3)
    
    # 1D trend filter: price above/below 100 EMA
    close_series = pd.Series(close)
    ema_100 = close_series.ewm(span=100, min_periods=100, adjust=False).mean().values
    
    # Volume filter: volume > 1.3x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema_100[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price rises above H3 or trend turns bearish or volume drops
            if (high[i] > h3_aligned[i] or close[i] < ema_100[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price falls below L3 or trend turns bullish or volume drops
            if (low[i] < l3_aligned[i] or close[i] > ema_100[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price reverses up from L3 with volume and bullish trend
            if ((low[i] <= l3_aligned[i] or close[i] <= l3_aligned[i]) and 
                close[i] > ema_100[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price reverses down from H3 with volume and bearish trend
            elif ((high[i] >= h3_aligned[i] or close[i] >= h3_aligned[i]) and 
                  close[i] < ema_100[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals