#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1D Donchian Breakout + Weekly Trend + Volume Confirmation
# Hypothesis: Daily Donchian(20) breakouts filtered by weekly trend direction and volume
# capture momentum in both bull and bear markets. Weekly trend filter avoids counter-trend
# entries during strong reversals, while volume ensures breakout validity.
# Target: 20-50 total trades over 4 years (5-12/year) to minimize fee drag on 1d.
name = "1d_donchian_breakout_weekly_trend_volume_v1"
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
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on daily data
    # Upper band = max(high, lookback=20)
    # Lower band = min(low, lookback=20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA(20) for trend filter
    weekly_close = df_weekly['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=20, adjust=False).mean().values
    weekly_ema_daily = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(weekly_ema_daily[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or weekly trend turns bearish
            if close[i] < donchian_low[i] or close[i] < weekly_ema_daily[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or weekly trend turns bullish
            if close[i] > donchian_high[i] or close[i] > weekly_ema_daily[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Long breakout: price breaks above Donchian high with bullish weekly trend
                if close[i] > donchian_high[i] and close[i] > weekly_ema_daily[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price breaks below Donchian low with bearish weekly trend
                elif close[i] < donchian_low[i] and close[i] < weekly_ema_daily[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals