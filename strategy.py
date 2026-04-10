#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 12h trend filter + volume confirmation
# - Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
# - Long when Bull Power > 0 AND Bear Power < 0 AND 12h close > 12h EMA50 AND volume > 1.5x average
# - Short when Bear Power > 0 AND Bull Power < 0 AND 12h close < 12h EMA50 AND volume > 1.5x average
# - Exit when power signals reverse OR volume drops below 0.8x average
# - Uses 12h trend filter to avoid counter-trend trades and volume spike for confirmation
# - Target: 12-30 trades/year (50-120 total over 4 years) to minimize fee drag
# - Works in bull markets via trend alignment, in bear via short signals from Bear Power dominance

name = "6h_12h_elder_ray_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Pre-compute Elder Ray components from 6h data
    # Bull Power = High - EMA13(close)
    # Bear Power = EMA13(close) - Low
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # EMA13 for Elder Ray calculation
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute volume filter: < 0.8x average volume for exit (loss of momentum)
    vol_weak = prices['volume'] < (0.8 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute aligned 12h data properly
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    # Align them to 6h timeframe
    h_12h_aligned = align_htf_to_ltf(prices, df_12h, h_12h)
    l_12h_aligned = align_htf_to_ltf(prices, df_12h, l_12h)
    c_12h_aligned = align_htf_to_ltf(prices, df_12h, c_12h)
    
    # Pre-compute 12h EMA(50) for trend filter
    ema50_12h = pd.Series(c_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(h_12h_aligned[i]) or np.isnan(l_12h_aligned[i]) or np.isnan(c_12h_aligned[i])):
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
            
            if prev_12h_idx >= 0 and not (np.isnan(h_12h_aligned[prev_12h_idx]) or 
                                         np.isnan(l_12h_aligned[prev_12h_idx]) or 
                                         np.isnan(c_12h_aligned[prev_12h_idx])):
                ph = h_12h_aligned[prev_12h_idx]  # Previous 12h period's high
                pl = l_12h_aligned[prev_12h_idx]  # Previous 12h period's low
                pc = c_12h_aligned[prev_12h_idx]  # Previous 12h period's close
                
                if position == 0:  # Flat - look for new entries
                    # Long entry: Bull Power > 0 AND Bear Power < 0 (bullish momentum) 
                    # AND 12h uptrend AND volume spike
                    if (bull_power[i] > 0 and bear_power[i] < 0 and 
                        close[i] > ema50_12h_aligned[i] and 
                        vol_spike.iloc[i]):
                        position = 1
                        signals[i] = 0.25
                    # Short entry: Bear Power > 0 AND Bull Power < 0 (bearish momentum)
                    # AND 12h downtrend AND volume spike
                    elif (bear_power[i] > 0 and bull_power[i] < 0 and 
                          close[i] < ema50_12h_aligned[i] and 
                          vol_spike.iloc[i]):
                        position = -1
                        signals[i] = -0.25
                else:  # Have position - look for exit
                    # Exit conditions:
                    # 1. Power signals reverse (loss of momentum)
                    # 2. Volume drops below 0.8x average (loss of conviction)
                    if position == 1:  # Long position
                        if (bull_power[i] <= 0 or bear_power[i] >= 0 or 
                            vol_weak.iloc[i]):
                            position = 0
                            signals[i] = 0.0
                        else:
                            signals[i] = 0.25  # Hold long
                    elif position == -1:  # Short position
                        if (bear_power[i] <= 0 or bull_power[i] >= 0 or 
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