#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 level with volume > 2.0x 20-bar average AND 1w close > 1w EMA20
# - Short when price breaks below Camarilla L3 level with volume > 2.0x 20-bar average AND 1w close < 1w EMA20
# - Exit when price retreats to Camarilla H4/L4 levels OR volume drops below 0.7x average
# - Uses 1w trend filter to avoid counter-trend trades and focus on primary trend
# - Higher volume threshold (2.0x) reduces trade frequency (target: 8-15 trades/year)
# - Tight exit on H4/L4 levels to capture quick reversals and limit drawdown
# - Designed to work in both bull (trend continuation) and bear (mean reversion at extremes) markets

name = "1d_1w_camarilla_breakout_volume_trend_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute volume confirmation: > 2.0x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (2.0 * volume_20_avg)
    
    # Pre-compute volume filter: < 0.7x average volume for exit (loss of momentum)
    vol_weak = prices['volume'] < (0.7 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute aligned 1w data properly
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    # Align them to 1d timeframe
    h_1w_aligned = align_htf_to_ltf(prices, df_1w, h_1w)
    l_1w_aligned = align_htf_to_ltf(prices, df_1w, l_1w)
    c_1w_aligned = align_htf_to_ltf(prices, df_1w, c_1w)
    
    # Pre-compute 1w EMA(20) for trend filter
    ema20_1w = pd.Series(c_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(h_1w_aligned[i]) or np.isnan(l_1w_aligned[i]) or np.isnan(c_1w_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get previous completed 1w bar values
        # Since 1d timeframe, 1w data updates every 5 bars (approx)
        # We use the aligned arrays directly - each 1w value repeats for 5 days
        # To ensure we're using completed 1w bar data, we check if we're at a weekly boundary
        
        # For 1d timeframe, we can use the aligned arrays directly
        # The aligned 1w data is already stretched to 1d resolution
        # We need to ensure we're using completed 1w bar data
        
        # Simple approach: use current aligned values as they represent the last completed 1w bar
        # align_htf_to_ltf already handles the delay for completed bars
        ph = h_1w_aligned[i]   # Current aligned 1w high (represents last completed 1w)
        pl = l_1w_aligned[i]   # Current aligned 1w low
        pc = c_1w_aligned[i]   # Current aligned 1w close
        
        # Calculate Camarilla levels from previous 1w bar
        range_val = ph - pl
        if range_val > 0:
            camarilla_h3 = pc + (range_val * 1.1 / 4)
            camarilla_l3 = pc - (range_val * 1.1 / 4)
            camarilla_h4 = pc + (range_val * 1.1 / 2)
            camarilla_l4 = pc - (range_val * 1.1 / 2)
            
            if position == 0:  # Flat - look for new breakout entries
                # Long breakout: price > Camarilla H3 with volume spike AND 1w uptrend
                if (prices['close'].iloc[i] > camarilla_h3 and 
                    vol_spike.iloc[i] and 
                    prices['close'].iloc[i] > ema20_1w_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Short breakdown: price < Camarilla L3 with volume spike AND 1w downtrend
                elif (prices['close'].iloc[i] < camarilla_l3 and 
                      vol_spike.iloc[i] and 
                      prices['close'].iloc[i] < ema20_1w_aligned[i]):
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
    
    return signals