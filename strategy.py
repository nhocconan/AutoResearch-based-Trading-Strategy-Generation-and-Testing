#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with volume confirmation and 1d trend filter
# - Long when price breaks above Camarilla H3 level with volume > 1.8x 20-bar average AND 1d close > 1d EMA50
# - Short when price breaks below Camarilla L3 level with volume > 1.8x 20-bar average AND 1d close < 1d EMA50
# - Exit when price retreats to Camarilla H4/L4 levels OR volume drops below 0.8x average
# - Uses 1d trend filter to avoid counter-trend trades in bear markets (2025+)
# - Moderate volume threshold (1.8x) balances signal quality and trade frequency (target: 12-25 trades/year)
# - Focus on BTC/ETH; SOL-only strategies are low value and will be discarded

name = "12h_1d_camarilla_breakout_volume_trend_v4"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute volume confirmation: > 1.8x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.8 * volume_20_avg)
    
    # Pre-compute volume filter: < 0.8x average volume for exit (loss of momentum)
    vol_weak = prices['volume'] < (0.8 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute aligned 1d data properly
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Align them to 12h timeframe
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
        
        # For 12h timeframe, 1d data updates every 2 bars
        # We need to use the previous completed 1d bar's data to avoid look-ahead
        # Since 1d data is aligned to 12h, each 1d value repeats for 2 consecutive 12h bars
        # To get the previous completed 1d bar, we look back to the start of the current 1d bar period
        
        # Calculate which 12h bar index corresponds to the start of the current 1d bar
        # 1d bar = 2 consecutive 12h bars
        # If i is even, we're at the start of a 1d bar (0, 2, 4, ...)
        # If i is odd, we're at the second 12h bar of a 1d bar (1, 3, 5, ...)
        if i % 2 == 0:
            # Current bar is start of 1d bar, so previous completed 1d bar started at i-2
            lookback_idx = i - 2
        else:
            # Current bar is second half of 1d bar, so previous completed 1d bar started at i-1
            lookback_idx = i - 1
        
        if lookback_idx >= 0:
            # Get the 1d bar data (high, low, close) from the aligned arrays
            # These values are constant across the 2 bars that make up the 1d bar
            ph = h_1d_aligned[lookback_idx]
            pl = l_1d_aligned[lookback_idx]
            pc = c_1d_aligned[lookback_idx]
            
            # Calculate Camarilla levels using previous completed 1d bar
            range_val = ph - pl
            if range_val > 0:
                camarilla_h3 = pc + (range_val * 1.1 / 4)
                camarilla_l3 = pc - (range_val * 1.1 / 4)
                camarilla_h4 = pc + (range_val * 1.1 / 2)
                camarilla_l4 = pc - (range_val * 1.1 / 2)
                
                if position == 0:  # Flat - look for new breakout entries
                    # Long breakout: price > Camarilla H3 with volume spike AND 1d uptrend
                    if (prices['close'].iloc[i] > camarilla_h3 and 
                        vol_spike.iloc[i] and 
                        prices['close'].iloc[i] > ema50_1d_aligned[i]):
                        position = 1
                        signals[i] = 0.25
                    # Short breakdown: price < Camarilla L3 with volume spike AND 1d downtrend
                    elif (prices['close'].iloc[i] < camarilla_l3 and 
                          vol_spike.iloc[i] and 
                          prices['close'].iloc[i] < ema50_1d_aligned[i]):
                        position = -1
                        signals[i] = -0.25
                    else:
                        # No entry signal, remain flat
                        signals[i] = 0.0
                else:  # Have position - look for exit
                    # Exit conditions:
                    # 1. Price retreats to Camarilla H4/L4 levels
                    # 2. Volume drops below 0.8x average (loss of momentum)
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
                # No valid range, hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # Not enough data yet for previous 1d bar, hold flat
            signals[i] = 0.0
    
    return signals