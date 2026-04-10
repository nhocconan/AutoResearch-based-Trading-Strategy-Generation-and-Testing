#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + ATR(21) trailing stop
# - Entry Long: Close > Donchian High(20) + 1d volume > 2.0x 20-period average
# - Entry Short: Close < Donchian Low(20) + 1d volume > 2.0x 20-period average
# - Exit: ATR(21) trailing stop (2.5x) on 4h timeframe
# - Position sizing: 0.25 (discrete levels to minimize fee churn)
# - Uses 1d volume for confirmation to avoid lower timeframe noise
# - Donchian channels provide clear structure, volume confirms breakout strength,
#   ATR trailing stop manages risk in volatile markets
# - Target: 15-40 trades/year (60-160 total over 4 years) to stay within HARD MAX: 400 total

name = "4h_1d_donchian_volume_trailing_v1"
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
    
    # Pre-compute 1d volume for confirmation
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian channels on 4h
    # Donchian High(20): highest high over past 20 periods
    # Donchian Low(20): lowest low over past 20 periods
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume spike confirmation
    # 20-period average volume
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_ma_20_1d * 2.0  # Volume > 2x average
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_threshold)
    
    # Align Donchian channels to 4h timeframe (already on 4h, but using helper for consistency)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
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
            np.isnan(volume_spike_aligned[i]) or np.isnan(atr_4h[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get current 4h close
        close_price = close_4h[i]
        
        # Volume spike confirmation: current 1d volume > 2x 20-period average
        # Need to get current 1d volume aligned to 4h
        volume_1d_current = df_1d['volume'].values
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_current)
        volume_confirmation = volume_1d_aligned[i] > volume_spike_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: close > Donchian High + volume confirmation
            if close_price > donchian_high_aligned[i] and volume_confirmation:
                position = 1
                highest_since_entry = prices['high'].iloc[i]
                signals[i] = 0.25
            # Short entry: close < Donchian Low + volume confirmation
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
                # ATR trailing stop: exit when price drops 2.5*ATR from highest point
                trailing_stop = prices['close'].iloc[i] < highest_since_entry - 2.5 * atr_4h[i]
                exit_condition = trailing_stop
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                # ATR trailing stop: exit when price rises 2.5*ATR from lowest point
                trailing_stop = prices['close'].iloc[i] > lowest_since_entry + 2.5 * atr_4h[i]
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