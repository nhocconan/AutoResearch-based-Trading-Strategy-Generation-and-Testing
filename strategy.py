#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and 1w trend filter
# - Long when price breaks above 20-period Donchian high AND 1d volume > 1.5x 20-period volume SMA AND price > 1w EMA(50)
# - Short when price breaks below 20-period Donchian low AND 1d volume > 1.5x 20-period volume SMA AND price < 1w EMA(50)
# - Exit: price returns to midpoint of Donchian channel or ATR-based trailing stop
# - Uses 1d for volume confirmation, 1w for trend filter, 12h for price action and Donchian channels
# - Volume spike ensures institutional participation; trend filter avoids counter-trend trades
# - Donchian channels provide objective breakout levels based on recent price action
# - Works in both bull and bear markets as breakouts occur in all regimes with volume confirmation

name = "12h_donchian_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Calculate 1d volume SMA for confirmation
    vol_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate ATR for stoploss (using 12h data)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Track highest high since entry for trailing stop (long) and lowest low for short
    highest_since_entry = np.full(n, np.nan)
    lowest_since_entry = np.full(n, np.nan)
    
    for i in range(20, n):  # Start from 20 to have sufficient lookback for indicators
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is invalid
        if (np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.5x 20-period volume SMA
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        vol_confirm = vol_1d_aligned[i] > 1.5 * volume_sma_20_1d_aligned[i]
        
        # Trend filter: price relative to 1w EMA(50)
        trend_filter_long = close[i] > ema_50_1w_aligned[i]
        trend_filter_short = close[i] < ema_50_1w_aligned[i]
        
        # Calculate 12h Donchian channels (20-period)
        if i >= 20:
            donchian_high = np.max(high[i-20:i])
            donchian_low = np.min(low[i-20:i])
            donchian_mid = (donchian_high + donchian_low) / 2.0
        else:
            donchian_high = np.max(high[:i+1])
            donchian_low = np.min(low[:i+1])
            donchian_mid = (donchian_high + donchian_low) / 2.0
        
        # Update highest high and lowest low since entry for trailing stop
        if position == 1:  # Long position
            if i == 20 or position == 0:  # New entry
                highest_since_entry[i] = high[i]
            else:
                highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
        elif position == -1:  # Short position
            if i == 20 or position == 0:  # New entry
                lowest_since_entry[i] = low[i]
            else:
                lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
        else:  # Flat
            highest_since_entry[i] = np.nan
            lowest_since_entry[i] = np.nan
        
        # Only trade when volume confirmation is present
        if vol_confirm:
            # Long breakout: price breaks above Donchian high AND trend filter long
            if close[i] > donchian_high and trend_filter_long:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25  # Maintain position
            # Short breakout: price breaks below Donchian low AND trend filter short
            elif close[i] < donchian_low and trend_filter_short:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25  # Maintain position
            # Exit conditions
            else:
                exit_signal = False
                # Exit long: price returns to midpoint or hits trailing stop
                if position == 1:
                    if close[i] <= donchian_mid or close[i] <= highest_since_entry[i] - 2.5 * atr[i]:
                        exit_signal = True
                # Exit short: price returns to midpoint or hits trailing stop
                elif position == -1:
                    if close[i] >= donchian_mid or close[i] >= lowest_since_entry[i] + 2.5 * atr[i]:
                        exit_signal = True
                
                if exit_signal:
                    if position != 0:  # Only signal on exit
                        position = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = 0.0  # Maintain flat
                else:
                    # Maintain current position
                    signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:
            # No volume confirmation: exit any position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals