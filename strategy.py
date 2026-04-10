#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h trend filter and volume confirmation
# - Long when price breaks above Donchian upper band with volume > 1.8x average AND 12h close > 12h EMA20
# - Short when price breaks below Donchian lower band with volume > 1.8x average AND 12h close < 12h EMA20
# - Exit when price retreats to Donchian midpoint OR volume drops below 1.0x average
# - Uses 12h trend filter to avoid counter-trend trades in bear markets (2025+)
# - Volume threshold (1.8x) reduces false breakouts and targets 12-30 trades/year (48-120 total over 4 years)
# - Higher timeframe (6h) reduces trade frequency vs 4h counterparts, lowering fee drag
# - Donchian channels provide clear structure for breakouts in both trending and ranging markets

name = "6h_12h_donchian_breakout_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Pre-compute 12h EMA(20) for trend filter
    close_12h = df_12h['close'].values
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # Pre-compute volume confirmation: > 1.8x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.8 * volume_20_avg)
    
    # Pre-compute volume filter: < 1.0x average volume for exit (loss of momentum)
    vol_weak = prices['volume'] < (1.0 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute Donchian channels from 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Align them to 6h timeframe
    high_12h_aligned = align_htf_to_ltf(prices, df_12h, high_12h)
    low_12h_aligned = align_htf_to_ltf(prices, df_12h, low_12h)
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema20_12h_aligned[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(high_12h_aligned[i]) or np.isnan(low_12h_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get previous completed 12h bar values (need to shift by 2 to avoid look-ahead)
        # Since 6h timeframe, there are 2 bars per 12h bar
        if i >= 4:  # Need at least 4 6h bars (2x 12h bars) to get previous 12h bar's data
            # Get index of previous completed 12h bar
            prev_12h_idx = i - 2  # Look back 2 bars (one 12h period)
            
            if prev_12h_idx >= 0 and not (np.isnan(high_12h_aligned[prev_12h_idx]) or 
                                         np.isnan(low_12h_aligned[prev_12h_idx])):
                ph = high_12h_aligned[prev_12h_idx]  # Previous 12h period's high
                pl = low_12h_aligned[prev_12h_idx]   # Previous 12h period's low
                
                # Calculate Donchian levels
                upper_band = ph  # Donchian upper = previous period's high
                lower_band = pl  # Donchian lower = previous period's low
                mid_band = (ph + pl) / 2  # Donchian midpoint
                
                if position == 0:  # Flat - look for new breakout entries
                    # Long breakout: price > Donchian upper with volume spike AND 12h uptrend
                    if (prices['close'].iloc[i] > upper_band and 
                        vol_spike.iloc[i] and 
                        prices['close'].iloc[i] > ema20_12h_aligned[i]):
                        position = 1
                        signals[i] = 0.25
                    # Short breakdown: price < Donchian lower with volume spike AND 12h downtrend
                    elif (prices['close'].iloc[i] < lower_band and 
                          vol_spike.iloc[i] and 
                          prices['close'].iloc[i] < ema20_12h_aligned[i]):
                        position = -1
                        signals[i] = -0.25
                else:  # Have position - look for exit
                    # Exit conditions:
                    # 1. Price retreats to Donchian midpoint
                    # 2. Volume drops below 1.0x average (loss of momentum)
                    if position == 1:  # Long position
                        if (prices['close'].iloc[i] < mid_band or 
                            vol_weak.iloc[i]):
                            position = 0
                            signals[i] = 0.0
                        else:
                            signals[i] = 0.25  # Hold long
                    elif position == -1:  # Short position
                        if (prices['close'].iloc[i] > mid_band or 
                            vol_weak.iloc[i]):
                            position = 0
                            signals[i] = 0.0
                        else:
                            signals[i] = -0.25  # Hold short
            else:
                # Hold current position
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