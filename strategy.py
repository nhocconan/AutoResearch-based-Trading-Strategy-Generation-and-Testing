#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Weekly Pivot Breakout with Volume Filter
# Hypothesis: Weekly pivot levels (R1/S1, R2/S2) act as strong support/resistance.
# Breakouts above R2 with volume indicate bullish continuation.
# Breakdowns below S2 with volume indicate bearish continuation.
# Uses 1w trend filter (price above/below 200 EMA) to avoid counter-trend trades.
# Volume filter ensures institutional participation. Works in bull/bear markets by
# aligning with trend: in bull, only long breakouts; in bear, only short breakdowns.
# Target: 5-25 trades/year (20-100 over 4 years).

name = "1d_weekly_pivot_breakout_volume_v1"
timeframe = "1d"
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
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly data (previous week's OHLC)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Shift by 1 to use previous week's data (avoid look-ahead)
    prev_weekly_high = np.roll(weekly_high, 1)
    prev_weekly_low = np.roll(weekly_low, 1)
    prev_weekly_close = np.roll(weekly_close, 1)
    prev_weekly_high[0] = prev_weekly_high[1] if len(prev_weekly_high) > 1 else 0
    prev_weekly_low[0] = prev_weekly_low[1] if len(prev_weekly_low) > 1 else 0
    prev_weekly_close[0] = prev_weekly_close[1] if len(prev_weekly_close) > 1 else 0
    
    # Calculate weekly pivot points
    weekly_range = prev_weekly_high - prev_weekly_low
    weekly_pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    weekly_r1 = weekly_pivot + (weekly_range * 1.0 / 2)
    weekly_s1 = weekly_pivot - (weekly_range * 1.0 / 2)
    weekly_r2 = weekly_pivot + weekly_range
    weekly_s2 = weekly_pivot - weekly_range
    
    # Align to 1d timeframe (use previous week's levels)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s2)
    
    # 1w trend filter: price above/below 200 EMA
    close_series = pd.Series(close)
    ema_200 = close_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Volume filter: volume > 2x 50-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=50, min_periods=50).mean().values
    vol_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(weekly_r2_aligned[i]) or np.isnan(weekly_s2_aligned[i]) or 
            np.isnan(ema_200[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below weekly S2 or trend turns bearish or volume drops
            if (close[i] < weekly_s2_aligned[i] or close[i] < ema_200[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above weekly R2 or trend turns bullish or volume drops
            if (close[i] > weekly_r2_aligned[i] or close[i] > ema_200[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above weekly R2 with volume and bullish trend
            if ((high[i] > weekly_r2_aligned[i] or close[i] > weekly_r2_aligned[i]) and 
                close[i] > ema_200[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below weekly S2 with volume and bearish trend
            elif ((low[i] < weekly_s2_aligned[i] or close[i] < weekly_s2_aligned[i]) and 
                  close[i] < ema_200[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals