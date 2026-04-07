#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: Daily Donchian Breakout with Weekly Trend Filter and Volume Confirmation
# Hypothesis: Breakouts from daily Donchian channels (20-period) occur with high momentum when aligned with weekly trend.
# Uses weekly EMA for trend filter and volume spike for confirmation. Works in both bull and bear by following higher-timeframe trend.
# Target: 20-50 trades/year (80-200 total over 4 years) to avoid excessive fee drag.

name = "1d_donchian_breakout_1w_ema_trend_v2"
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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    
    # Daily Donchian Channel (20-period)
    donchian_period = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(donchian_period - 1, n):
        upper[i] = np.max(high[i - donchian_period + 1:i + 1])
        lower[i] = np.min(low[i - donchian_period + 1:i + 1])
    
    # Weekly EMA(30) for trend filter
    ema_30_weekly = pd.Series(close_weekly).ewm(span=30, adjust=False).mean().values
    ema_30_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_30_weekly)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(donchian_period, n):
        # Skip if required data not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(ema_30_weekly_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price closes below weekly EMA (trend change) or Donchian lower band
            if close[i] < ema_30_weekly_aligned[i] or close[i] < lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above weekly EMA (trend change) or Donchian upper band
            if close[i] > ema_30_weekly_aligned[i] or close[i] > upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Breakout above upper band with uptrend
                if close[i] > upper[i] and close[i] > ema_30_weekly_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakdown below lower band with downtrend
                elif close[i] < lower[i] and close[i] < ema_30_weekly_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals