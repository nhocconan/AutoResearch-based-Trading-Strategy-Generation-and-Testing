#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Monthly Donchian Breakout with Volume Filter
# Hypothesis: Monthly Donchian(20) breakouts with volume confirmation capture
# strong momentum moves while avoiding false breakouts. Monthly timeframe
# provides robust trend across bull/bear cycles, volume filter reduces false signals.
# Target: 20-30 trades/year (80-120 total over 4 years) to minimize fee drag.

name = "4h_monthly_donchian_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get monthly data for trend and breakout levels
    df_monthly = get_htf_data(prices, '1M')
    if len(df_monthly) < 30:
        return np.zeros(n)
    
    # Calculate monthly EMA(50) for trend filter
    monthly_close = df_monthly['close'].values
    ema_50 = pd.Series(monthly_close).ewm(span=50, adjust=False).mean().values
    
    # Monthly Donchian channels (20-period high/low)
    monthly_high = df_monthly['high'].values
    monthly_low = df_monthly['low'].values
    high_series = pd.Series(monthly_high)
    low_series = pd.Series(monthly_low)
    monthly_high_20 = high_series.rolling(window=20, min_periods=20).max().values
    monthly_low_20 = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align monthly indicators to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_monthly, ema_50)
    high_20_aligned = align_htf_to_ltf(prices, df_monthly, monthly_high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_monthly, monthly_low_20)
    
    # Volume filter on 4h: volume > 1.3x 30-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if required data not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(high_20_aligned[i]) or
            np.isnan(low_20_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls back below 20-period low or closes below EMA50
            if close[i] < low_20_aligned[i] or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises back above 20-period high or closes above EMA50
            if close[i] > high_20_aligned[i] or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Strong trend filter: price must be away from EMA50
            if close[i] > ema_50_aligned[i] * 1.02:  # Uptrend: price > EMA50 by 2%
                # Long entry: breakout above 20-period high with volume
                if (high[i] > high_20_aligned[i] and close[i] > high_20_aligned[i] and
                    vol_filter[i]):
                    position = 1
                    signals[i] = 0.25
            elif close[i] < ema_50_aligned[i] * 0.98:  # Downtrend: price < EMA50 by 2%
                # Short entry: breakdown below 20-period low with volume
                if (low[i] < low_20_aligned[i] and close[i] < low_20_aligned[i] and
                      vol_filter[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals