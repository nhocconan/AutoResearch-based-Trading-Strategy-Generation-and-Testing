#!/usr/bin/env python3
# 12h_camarilla_pivot_volume_spike_v1
# Hypothesis: 12h Camarilla pivot level touch with volume spike and daily trend filter.
# Works in bull markets by buying dips to support in uptrend and selling rallies to resistance in downtrend.
# In bear markets, captures mean-reversion at extreme levels when price reverts to mean from overbought/oversold.
# Uses Camarilla levels (H3/L3) from daily timeframe for institutional support/resistance.
# Volume spike confirms institutional participation. Target: 15-25 trades/year with ~0.25 position size.

name = "12h_camarilla_pivot_volume_spike_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume moving average (20-period)
    vol_ma = np.zeros_like(volume)
    vol_ma[19:] = np.convolve(volume, np.ones(20)/20, mode='valid')
    vol_ma[:19] = vol_ma[19]
    
    # Get daily data for Camarilla levels
    df_daily = get_htf_data(prices, '1d')
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate Camarilla levels for each day
    # H3/L3 are the key levels for intraday trading
    # H3 = Close + 1.1 * (High - Low) / 2
    # L3 = Close - 1.1 * (High - Low) / 2
    camarilla_h3 = close_daily + 1.1 * (high_daily - low_daily) / 2
    camarilla_l3 = close_daily - 1.1 * (high_daily - low_daily) / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_l3)
    
    # Daily EMA (50-period) for trend filter
    ema_period = 50
    ema_daily = np.zeros_like(close_daily)
    if len(close_daily) >= ema_period:
        ema_daily[ema_period-1] = np.mean(close_daily[:ema_period])
        for i in range(ema_period, len(close_daily)):
            ema_daily[i] = (close_daily[i] * 2 + ema_daily[i-1] * (ema_period - 1)) / (ema_period + 1)
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = max(20, ema_period) + 5
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(vol_ma[i]) or volume[i] == 0 or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(ema_daily_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8x 20-period average
        volume_filter = volume[i] > 1.8 * vol_ma[i]
        
        # Daily trend filter
        uptrend_htf = close[i] > ema_daily_aligned[i]
        downtrend_htf = close[i] < ema_daily_aligned[i]
        
        if position == 1:  # Long position
            # Exit if price reaches H3 (resistance) or trend fails
            if close[i] >= camarilla_h3_aligned[i] or not uptrend_htf or not volume_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price reaches L3 (support) or trend fails
            if close[i] <= camarilla_l3_aligned[i] or not downtrend_htf or not volume_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price near L3 support with volume spike and uptrend
            if (close[i] <= camarilla_l3_aligned[i] * 1.005) and volume_filter and uptrend_htf:
                position = 1
                signals[i] = 0.25
            # Short entry: price near H3 resistance with volume spike and downtrend
            elif (close[i] >= camarilla_h3_aligned[i] * 0.995) and volume_filter and downtrend_htf:
                position = -1
                signals[i] = -0.25
    
    return signals