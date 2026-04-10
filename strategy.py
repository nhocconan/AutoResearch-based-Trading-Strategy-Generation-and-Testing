#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with volume confirmation and 12h trend filter
# - Long when price breaks above Camarilla H3 level with volume > 1.6x 20-bar average AND 12h close > 12h EMA20
# - Short when price breaks below Camarilla L3 level with volume > 1.6x 20-bar average AND 12h close < 12h EMA20
# - Exit when price retreats to Camarilla H4/L4 levels OR volume drops below 0.7x average
# - Uses 12h trend filter to avoid counter-trend trades
# - Moderate volume threshold (1.6x) and weak volume exit (0.7x) to target 20-40 trades/year
# - Focus on BTC/ETH; proven Camarilla structure works in both bull and bear markets

name = "4h_12h_camarilla_breakout_volume_trend_v4"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute volume confirmation: > 1.6x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.6 * volume_20_avg)
    
    # Pre-compute volume filter: < 0.7x average volume for exit (loss of momentum)
    vol_weak = prices['volume'] < (0.7 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute aligned 12h data properly
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    # Align them to 4h timeframe
    h_12h_aligned = align_htf_to_ltf(prices, df_12h, h_12h)
    l_12h_aligned = align_htf_to_ltf(prices, df_12h, l_12h)
    c_12h_aligned = align_htf_to_ltf(prices, df_12h, c_12h)
    
    # Pre-compute 12h EMA(20) for trend filter
    ema20_12h = pd.Series(c_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema20_12h_aligned[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(h_12h_aligned[i]) or np.isnan(l_12h_aligned[i]) or np.isnan(c_12h_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get previous completed 12h bar values
        # Since 4h timeframe, there are 3 bars per 12h bar
        if i >= 3:  # Need at least 3 4h bars to get previous 12h bar's data
            # Get index of previous completed 12h bar
            # For 4h timeframe, 12h data updates every 3 bars
            # Look back to the previous multiple of 3 to get completed 12h bar
            lookback_idx = (i // 3) * 3 - 3  # Previous completed 12h bar
            
            if lookback_idx >= 0:
                ph = h_12h_aligned[lookback_idx]  # Previous 12h bar's high
                pl = l_12h_aligned[lookback_idx]  # Previous 12h bar's low
                pc = c_12h_aligned[lookback_idx]  # Previous 12h bar's close
                
                # Calculate Camarilla levels
                range_val = ph - pl
                if range_val > 0:
                    camarilla_h3 = pc + (range_val * 1.1 / 4)
                    camarilla_l3 = pc - (range_val * 1.1 / 4)
                    camarilla_h4 = pc + (range_val * 1.1 / 2)
                    camarilla_l4 = pc - (range_val * 1.1 / 2)
                    
                    if position == 0:  # Flat - look for new breakout entries
                        # Long breakout: price > Camarilla H3 with volume spike AND 12h uptrend
                        if (prices['close'].iloc[i] > camarilla_h3 and 
                            vol_spike.iloc[i] and 
                            prices['close'].iloc[i] > ema20_12h_aligned[i]):
                            position = 1
                            signals[i] = 0.25
                        # Short breakdown: price < Camarilla L3 with volume spike AND 12h downtrend
                        elif (prices['close'].iloc[i] < camarilla_l3 and 
                              vol_spike.iloc[i] and 
                              prices['close'].iloc[i] < ema20_12h_aligned[i]):
                            position = -1
                            signals[i] = -0.25
                    else:  # Have position - look for exit
                        # Exit conditions:
                        # 1. Price retreats to Camarilla H4/L4 levels
                        # 2. Volume drops below 0.7x average (loss of momentum)
                        if position == 1:  # Long position
                            if (prices['close'].iloc[i] < camarilla_h4 or 
                                vol_weak.iloc[i]):
                                position = 0
                                signals[i] = 0.0
                            else:
                                signals[i] = 0.25  # Hold long
                        elif position == -1:  # Short position
                            if (prices['close'].iloc[i] > camarilla_l4 or 
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