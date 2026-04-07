#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian 20 with Weekly Trend Filter and Volume Confirmation
# Hypothesis: Price breaking above/below 20-period Donchian channel on 12h timeframe,
# confirmed by weekly trend (price above/below 50-period EMA) and volume spike,
# captures institutional breakouts that persist. Works in bull (breakouts continue)
# and bear (breakdowns continue) as institutional activity drives trends.
# Target: 15-30 trades/year (60-120 over 4 years) to avoid fee drag.

name = "12h_donchian20_weekly_trend_volume_v1"
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
    
    # Get weekly data for trend filter (EMA50)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    weekly_close = df_weekly['close'].values
    ema50_weekly = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Calculate 12h Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 2.0x 20-period average (strict to reduce trades)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_weekly_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below Donchian low or weekly trend turns bearish
            if (close[i] <= donchian_low[i] or 
                close[i] < ema50_weekly_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above Donchian high or weekly trend turns bullish
            if (close[i] >= donchian_high[i] or 
                close[i] > ema50_weekly_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above Donchian high with volume and bullish weekly trend
            if (high[i] > donchian_high[i] and 
                close[i] > donchian_high[i] and 
                vol_filter[i] and 
                close[i] > ema50_weekly_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low with volume and bearish weekly trend
            elif (low[i] < donchian_low[i] and 
                  close[i] < donchian_low[i] and 
                  vol_filter[i] and 
                  close[i] < ema50_weekly_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals