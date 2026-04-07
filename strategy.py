#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Donchian Breakout + Weekly Trend + Volume Confirmation
# Hypothesis: Weekly trend filters Donchian(20) breakouts on daily to avoid false signals.
# Breakouts aligned with weekly trend capture sustained moves; counter-trend breakouts faded.
# Volume confirmation ensures institutional participation. Works in bull (trend-following) and bear (mean-reversion at extremes).
# Target: 15-25 trades/year (60-100 total over 4 years).

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
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    
    # Weekly EMA(20) for trend filter
    ema_20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False).mean().values
    ema_20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_20_weekly)
    
    # Daily Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_20_weekly_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price closes below weekly EMA or Donchian lower band
            if close[i] < ema_20_weekly_aligned[i] or close[i] < low_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above weekly EMA or Donchian upper band
            if close[i] > ema_20_weekly_aligned[i] or close[i] > high_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # In uptrend (price > weekly EMA): buy breakout above upper band
                if close[i] > ema_20_weekly_aligned[i] and close[i] > high_20[i]:
                    position = 1
                    signals[i] = 0.25
                # In downtrend (price < weekly EMA): sell breakdown below lower band
                elif close[i] < ema_20_weekly_aligned[i] and close[i] < low_20[i]:
                    position = -1
                    signals[i] = -0.25
                # In ranging market (price near weekly EMA): fade at extremes
                elif abs(close[i] - ema_20_weekly_aligned[i]) / ema_20_weekly_aligned[i] < 0.01:
                    # Buy near lower band, sell near upper band
                    if close[i] < low_20[i] * 1.005:  # Near lower band
                        position = 1
                        signals[i] = 0.25
                    elif close[i] > high_20[i] * 0.995:  # Near upper band
                        position = -1
                        signals[i] = -0.25
    
    return signals