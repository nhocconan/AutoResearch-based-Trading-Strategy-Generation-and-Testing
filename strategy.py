#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Daily Range Breakout with Weekly Trend and Volume Filter
# Hypothesis: On daily timeframe, price breaking above/below the prior day's high/low with
# volume confirmation and weekly trend alignment captures momentum moves. Weekly trend filter
# prevents counter-trend trades in strong trends. Works in both bull and bear markets by
# following the weekly trend direction. Target: 15-25 trades/year (60-100 over 4 years).
name = "1d_daily_range_breakout_weekly_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
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
    
    # Weekly EMA(20) for trend filter
    weekly_close = df_weekly['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=20, adjust=False).mean().values
    weekly_ema_1d = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    
    # Previous day's high and low (shifted by 1 to avoid look-ahead)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = np.nan  # First bar has no previous day
    prev_low[0] = np.nan
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or 
            np.isnan(weekly_ema_1d[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below previous day's low (stop/reversal)
            if close[i] < prev_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above previous day's high (stop/reversal)
            if close[i] > prev_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Long: price breaks above previous day's high with weekly uptrend
                if close[i] > prev_high[i] and close[i] > weekly_ema_1d[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below previous day's low with weekly downtrend
                elif close[i] < prev_low[i] and close[i] < weekly_ema_1d[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals