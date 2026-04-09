#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA trend filter + volume confirmation
# - Uses 1d Donchian channel breakouts for trend-following entries
# - Uses 1w EMA(50) to filter regime: only trade in direction of weekly trend
# - Requires volume > 1.5x 20-period average for confirmation
# - Long when price breaks above Donchian upper band AND weekly EMA rising
# - Short when price breaks below Donchian lower band AND weekly EMA falling
# - Fixed position size 0.25 to control drawdown
# - ATR-based stoploss: exit when price moves 2.5x ATR against position
# - Works in both bull and bear markets by aligning with weekly trend
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)

name = "1d_1w_donchian_ema_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ATR(14) for stoploss and volume average
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i]) or
            np.isnan(vol_ma[i]) or atr[i] <= 0 or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: rising EMA for long, falling EMA for short
        ema_rising = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]
        ema_falling = ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # ATR-based trailing stop: exit if price drops 2.5x ATR from highest high
            if close[i] < highest_high_since_entry - 2.5 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # ATR-based trailing stop: exit if price rises 2.5x ATR from lowest low
            if close[i] > lowest_low_since_entry + 2.5 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirm:
                # Long entry: price breaks above Donchian upper band AND weekly EMA rising
                if close[i] > highest_high[i] and ema_rising:
                    position = 1
                    highest_high_since_entry = high[i]
                    lowest_low_since_entry = low[i]
                    signals[i] = 0.25
                # Short entry: price breaks below Donchian lower band AND weekly EMA falling
                elif close[i] < lowest_low[i] and ema_falling:
                    position = -1
                    highest_high_since_entry = high[i]
                    lowest_low_since_entry = low[i]
                    signals[i] = -0.25
    
    return signals