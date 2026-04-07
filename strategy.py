#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with daily trend filter (EMA50) and volume confirmation
# Uses 12h price breaking above/below 20-period high/low for entry, confirmed by daily EMA50 trend
# and volume > 1.5x 20-period average. Exits when price returns to Donchian midpoint or on opposite signal.
# Designed for low trade frequency (target: 12-37/year) with volatility-adjusted sizing to manage risk.
# Works in bull markets via breakout momentum and in bear markets via breakdown continuation.

name = "12h_donchian20_daily_ema_volume_v1"
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
    
    # Daily data for EMA filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 12h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Calculate daily EMA50
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_avg[i]) or vol_avg[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Breakout conditions
        breakout_up = close[i] > highest_high[i] and vol_confirm
        breakout_down = close[i] < lowest_low[i] and vol_confirm
        
        # Trend filter from daily EMA50
        uptrend = close[i] > ema_1d_aligned[i]
        downtrend = close[i] < ema_1d_aligned[i]
        
        # Exit conditions: price returns to midpoint
        exit_long = close[i] < donchian_mid[i]
        exit_short = close[i] > donchian_mid[i]
        
        # Generate signals with hysteresis
        if i == 50:
            # Initial state
            if breakout_up and uptrend:
                signals[i] = 0.25
            elif breakout_down and downtrend:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:
            # Continue existing position until exit signal
            if signals[i-1] > 0:  # Currently long
                if exit_long:
                    signals[i] = 0.0
                else:
                    signals[i] = signals[i-1]  # Hold position
            elif signals[i-1] < 0:  # Currently short
                if exit_short:
                    signals[i] = 0.0
                else:
                    signals[i] = signals[i-1]  # Hold position
            else:  # Currently flat
                if breakout_up and uptrend:
                    signals[i] = 0.25
                elif breakout_down and downtrend:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
    
    return signals