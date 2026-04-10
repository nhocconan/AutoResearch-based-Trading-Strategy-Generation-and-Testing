#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d volume confirmation and ATR trailing stop
# - Entry: Long when price breaks above Donchian(20) high + 1d volume > 1.3x 20-period average
#          Short when price breaks below Donchian(20) low + 1d volume > 1.3x 20-period average
# - Exit: ATR(21) trailing stop (2.0x) on 4h timeframe
# - Position sizing: 0.25 (discrete levels to minimize fee churn)
# - Uses 1d for volume confirmation to avoid lower timeframe noise
# - Donchian breakouts capture strong trending moves, volume confirms participation
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within HARD MAX: 400 total

name = "4h_1d_donchian_volume_trailing_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Pre-compute 1d OHLC for volume
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian channels (20-period) on 4h
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Align Donchian channels to 4h (already on 4h, but ensure proper alignment)
    donchian_high_aligned = donchian_high  # Already on 4h timeframe
    donchian_low_aligned = donchian_low    # Already on 4h timeframe
    
    # Pre-compute 4h ATR(21) for trailing stop
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr_4h = pd.Series(tr).rolling(window=21, min_periods=21).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0  # for trailing stop
    lowest_since_entry = 0.0   # for trailing stop
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_4h[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get current 4h close
        close_price = close_4h[i]
        
        # Get current 1d volume for confirmation (need to align raw volume)
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirmation = volume_1d_aligned[i] > 1.3 * volume_ma_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high + volume confirmation
            if close_price > donchian_high_aligned[i] and volume_confirmation:
                position = 1
                highest_since_entry = prices['high'].iloc[i]
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low + volume confirmation
            elif close_price < donchian_low_aligned[i] and volume_confirmation:
                position = -1
                lowest_since_entry = prices['low'].iloc[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or trailing stop
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, prices['high'].iloc[i])
                # ATR trailing stop: exit when price drops 2.0*ATR from highest point
                trailing_stop = prices['close'].iloc[i] < highest_since_entry - 2.0 * atr_4h[i]
                exit_condition = trailing_stop
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                # ATR trailing stop: exit when price rises 2.0*ATR from lowest point
                trailing_stop = prices['close'].iloc[i] > lowest_since_entry + 2.0 * atr_4h[i]
                exit_condition = trailing_stop
            
            if exit_condition:
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals