#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Monthly Pivot Breakout with Volume and Trend Filter
# Hypothesis: Monthly pivot levels (R1/S1, R2/S2) act as strong support/resistance.
# Breakouts above R2 with volume and trend confirmation indicate bullish continuation.
# Breakdowns below S2 with volume and trend confirmation indicate bearish continuation.
# Uses 1d trend filter (price above/below 50 EMA) to avoid counter-trend trades.
# Volume filter ensures institutional participation. Monthly pivots are more stable
# and less prone to noise than daily pivots, reducing false breakouts.
# Target: 10-30 trades/year (40-120 over 4 years).

name = "6h_monthly_pivot_breakout_volume_trend_v1"
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
    
    # Get monthly data for pivot calculation
    df_monthly = get_htf_data(prices, '1M')
    if len(df_monthly) < 2:
        return np.zeros(n)
    
    # Calculate monthly data (previous month's OHLC)
    monthly_high = df_monthly['high'].values
    monthly_low = df_monthly['low'].values
    monthly_close = df_monthly['close'].values
    
    # Shift by 1 to use previous month's data (avoid look-ahead)
    prev_monthly_high = np.roll(monthly_high, 1)
    prev_monthly_low = np.roll(monthly_low, 1)
    prev_monthly_close = np.roll(monthly_close, 1)
    prev_monthly_high[0] = prev_monthly_high[1] if len(prev_monthly_high) > 1 else 0
    prev_monthly_low[0] = prev_monthly_low[1] if len(prev_monthly_low) > 1 else 0
    prev_monthly_close[0] = prev_monthly_close[1] if len(prev_monthly_close) > 1 else 0
    
    # Calculate monthly pivot points
    monthly_range = prev_monthly_high - prev_monthly_low
    monthly_pivot = (prev_monthly_high + prev_monthly_low + prev_monthly_close) / 3.0
    monthly_r1 = monthly_pivot + (monthly_range * 1.0 / 2)
    monthly_s1 = monthly_pivot - (monthly_range * 1.0 / 2)
    monthly_r2 = monthly_pivot + monthly_range
    monthly_s2 = monthly_pivot - monthly_range
    
    # Align to 6h timeframe (use previous month's levels)
    monthly_r2_aligned = align_htf_to_ltf(prices, df_monthly, monthly_r2)
    monthly_s2_aligned = align_htf_to_ltf(prices, df_monthly, monthly_s2)
    
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
        if (np.isnan(monthly_r2_aligned[i]) or np.isnan(monthly_s2_aligned[i]) or 
            np.isnan(ema_50[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls to S2 or trend turns bearish or volume drops
            if (close[i] <= monthly_s2_aligned[i] or close[i] < ema_50[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises to R2 or trend turns bullish or volume drops
            if (close[i] >= monthly_r2_aligned[i] or close[i] > ema_50[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above R2 with volume and bullish trend
            if ((high[i] > monthly_r2_aligned[i] or close[i] > monthly_r2_aligned[i]) and 
                close[i] > ema_50[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below S2 with volume and bearish trend
            elif ((low[i] < monthly_s2_aligned[i] or close[i] < monthly_s2_aligned[i]) and 
                  close[i] < ema_50[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals