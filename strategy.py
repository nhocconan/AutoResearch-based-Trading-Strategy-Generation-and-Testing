#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# - Long when price breaks above 20-bar high AND 1d close > 1d EMA50 AND volume > 1.5x 20-bar average
# - Short when price breaks below 20-bar low AND 1d close < 1d EMA50 AND volume > 1.5x 20-bar average
# - Exit when price retreats to opposite Donchian level (10-bar low for longs, 10-bar high for shorts)
# - Uses 1d trend filter to avoid counter-trend trades in bear markets (2025+)
# - Volume confirmation reduces false breakouts
# - Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag
# - Focus on BTC/ETH; proven pattern from DB top performers

name = "4h_1d_donchian_breakout_volume_trend_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute aligned 1d data properly
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Align them to 4h timeframe
    h_1d_aligned = align_htf_to_ltf(prices, df_1d, h_1d)
    l_1d_aligned = align_htf_to_ltf(prices, df_1d, l_1d)
    c_1d_aligned = align_htf_to_ltf(prices, df_1d, c_1d)
    
    # Pre-compute 1d EMA(50) for trend filter
    ema50_1d = pd.Series(c_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(h_1d_aligned[i]) or np.isnan(l_1d_aligned[i]) or np.isnan(c_1d_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Calculate Donchian channels (20-period for entry, 10-period for exit)
        if i >= 20:
            # 20-period high/low for entry signals
            high_20 = prices['high'].iloc[i-20:i].max()
            low_20 = prices['low'].iloc[i-20:i].min()
        else:
            # Not enough data for entry signals yet
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
            
        if i >= 10:
            # 10-period high/low for exit signals
            high_10 = prices['high'].iloc[i-10:i].max()
            low_10 = prices['low'].iloc[i-10:i].min()
        else:
            # Not enough data for exit signals yet, use 20-period
            high_10 = high_20
            low_10 = low_20
        
        # Get previous completed 1d bar values for trend filter
        # Since 4h timeframe, 1d data updates every 6 bars
        if i >= 6:
            # Look back to get completed 1d bar (multiple of 6 bars back)
            lookback_idx = i - (i % 6) - 6  # Previous completed 1d bar
            if lookback_idx >= 0:
                # Use aligned 1d data directly
                pc = c_1d_aligned[lookback_idx]
                ema50 = ema50_1d_aligned[lookback_idx]
                
                # Only proceed if we have valid data
                if not (np.isnan(pc) or np.isnan(ema50)):
                    if position == 0:  # Flat - look for new breakout entries
                        # Long breakout: price > 20-period high with volume spike AND 1d uptrend
                        if (prices['high'].iloc[i] > high_20 and 
                            vol_spike.iloc[i] and 
                            pc > ema50):
                            position = 1
                            signals[i] = 0.25
                        # Short breakdown: price < 20-period low with volume spike AND 1d downtrend
                        elif (prices['low'].iloc[i] < low_20 and 
                              vol_spike.iloc[i] and 
                              pc < ema50):
                            position = -1
                            signals[i] = -0.25
                        else:
                            signals[i] = 0.0  # Stay flat
                    else:  # Have position - look for exit
                        # Exit when price retreats to opposite Donchian level
                        if position == 1:  # Long position
                            if prices['low'].iloc[i] < low_10:
                                position = 0
                                signals[i] = 0.0
                            else:
                                signals[i] = 0.25  # Hold long
                        elif position == -1:  # Short position
                            if prices['high'].iloc[i] > high_10:
                                position = 0
                                signals[i] = 0.0
                            else:
                                signals[i] = -0.25  # Hold short
                else:
                    # Hold current position if data invalid
                    if position == 0:
                        signals[i] = 0.0
                    elif position == 1:
                        signals[i] = 0.25
                    else:
                        signals[i] = -0.25
            else:
                # Not enough 1d history yet, hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # Not enough data yet, hold flat
            signals[i] = 0.0
    
    return signals