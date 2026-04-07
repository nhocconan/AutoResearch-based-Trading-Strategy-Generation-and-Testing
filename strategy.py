#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Daily Donchian Breakout with Volume and EMA Filter
# Hypothesis: Breakouts of daily Donchian channels (20-period) on 4h timeframe
# with volume confirmation and EMA50 trend filter work in both bull and bear markets.
# In bull markets: buy breakouts above upper band with volume and price above EMA50.
# In bear markets: sell breakdowns below lower band with volume and price below EMA50.
# Target: 20-50 trades/year (80-200 over 4 years).

name = "4h_daily_donchian_breakout_volume_ema_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channel calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period)
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    
    # Upper and lower bands (no look-ahead: use previous day's data)
    upper_band = np.maximum.accumulate(daily_high)
    lower_band = np.minimum.accumulate(daily_low)
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    upper_band = np.roll(upper_band, 1)
    lower_band = np.roll(lower_band, 1)
    
    # Handle first element
    if len(upper_band) > 1:
        upper_band[0] = upper_band[1]
        lower_band[0] = lower_band[1]
    else:
        upper_band[0] = 0
        lower_band[0] = 0
    
    # Align to 4h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_daily, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_daily, lower_band)
    
    # Trend filter: EMA50
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or
            np.isnan(ema_50[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: breakdown below lower band or trend reversal
            if (low[i] < lower_band_aligned[i]) or (close[i] < ema_50[i]) or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit conditions: breakout above upper band or trend reversal
            if (high[i] > upper_band_aligned[i]) or (close[i] > ema_50[i]) or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: breakout above upper band with volume and trend filter
            if (high[i] > upper_band_aligned[i]) and vol_filter[i] and (close[i] > ema_50[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below lower band with volume and trend filter
            elif (low[i] < lower_band_aligned[i]) and vol_filter[i] and (close[i] < ema_50[i]):
                position = -1
                signals[i] = -0.25
    
    return signals