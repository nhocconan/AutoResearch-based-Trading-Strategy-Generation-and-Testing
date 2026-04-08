#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Daily Pivot Breakout with Volume Filter and Trend Filter
# Hypothesis: Daily pivot levels (R1/S1, R2/S2) act as intraday support/resistance.
# Breakouts above R2 with volume and trend confirmation indicate bullish continuation.
# Breakdowns below S2 with volume and trend confirmation indicate bearish continuation.
# Uses 1d trend filter (price above/below 50 EMA) to avoid counter-trend trades.
# Volume filter ensures institutional participation. Works in bull/bear markets by
# aligning with trend: in bull, only long breakouts; in bear, only short breakdowns.
# Target: 15-40 trades/year (60-160 over 4 years).

name = "6h_daily_pivot_breakout_volume_trend_v1"
timeframe = "6h"
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
    
    # Get daily data for pivot calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate daily data (previous day's OHLC)
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    prev_daily_high = np.roll(daily_high, 1)
    prev_daily_low = np.roll(daily_low, 1)
    prev_daily_close = np.roll(daily_close, 1)
    prev_daily_high[0] = prev_daily_high[1] if len(prev_daily_high) > 1 else 0
    prev_daily_low[0] = prev_daily_low[1] if len(prev_daily_low) > 1 else 0
    prev_daily_close[0] = prev_daily_close[1] if len(prev_daily_close) > 1 else 0
    
    # Calculate daily pivot points
    daily_range = prev_daily_high - prev_daily_low
    daily_pivot = (prev_daily_high + prev_daily_low + prev_daily_close) / 3.0
    daily_r1 = daily_pivot + (daily_range * 1.0 / 2)
    daily_s1 = daily_pivot - (daily_range * 1.0 / 2)
    daily_r2 = daily_pivot + daily_range
    daily_s2 = daily_pivot - daily_range
    
    # Align to 6h timeframe (use previous day's levels)
    daily_r2_aligned = align_htf_to_ltf(prices, df_daily, daily_r2)
    daily_s2_aligned = align_htf_to_ltf(prices, df_daily, daily_s2)
    
    # 1d trend filter: price above/below 50 EMA
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(daily_r2_aligned[i]) or np.isnan(daily_s2_aligned[i]) or 
            np.isnan(ema_50[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls to S1 or trend turns bearish or volume drops
            if (close[i] <= daily_s2_aligned[i] or close[i] < ema_50[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises to R1 or trend turns bullish or volume drops
            if (close[i] >= daily_r2_aligned[i] or close[i] > ema_50[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above R2 with volume and bullish trend
            if ((high[i] > daily_r2_aligned[i] or close[i] > daily_r2_aligned[i]) and 
                close[i] > ema_50[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below S2 with volume and bearish trend
            elif ((low[i] < daily_s2_aligned[i] or close[i] < daily_s2_aligned[i]) and 
                  close[i] < ema_50[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals