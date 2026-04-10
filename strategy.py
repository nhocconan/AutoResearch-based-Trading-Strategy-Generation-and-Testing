#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with volume confirmation and 1w trend filter
# - Long when price breaks above Camarilla H3 level with volume > 1.8x 20-bar average AND 1w close > 1w EMA50
# - Short when price breaks below Camarilla L3 level with volume > 1.8x 20-bar average AND 1w close < 1w EMA50
# - Exit when price retreats to Camarilla H4/L4 levels OR volume drops below 0.8x average
# - Uses 1w trend filter to avoid counter-trend trades in bear markets (2025+)
# - Moderate volume threshold (1.8x) balances signal quality and trade frequency (target: 20-50 trades/year)
# - Focus on BTC/ETH; SOL-only strategies are low value and will be discarded

name = "1d_1w_camarilla_breakout_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute volume confirmation: > 1.8x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.8 * volume_20_avg)
    
    # Pre-compute volume filter: < 0.8x average volume for exit (loss of momentum)
    vol_weak = prices['volume'] < (0.8 * volume_20_avg)
    
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
    
    # Pre-compute 1w EMA(50) for trend filter
    ema50_1w = pd.Series(c_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_20_avg[i]) or 
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
        # Since 1d timeframe, we look back 5 bars (5x 1d = 1w) for previous completed 1w bar
        if i >= 5:  # Need at least 5 1d bars to get previous 1w bar's data
            # Get index of previous completed 1w bar
            prev_1w_idx = i - 5  # Look back 5 bars (one 1w period = 5x 1d)
            
            if prev_1w_idx >= 0 and not (np.isnan(h_1w_aligned[prev_1w_idx]) or 
                                        np.isnan(l_1w_aligned[prev_1w_idx]) or 
                                        np.isnan(c_1w_aligned[prev_1w_idx])):
                ph = h_1w_aligned[prev_1w_idx]  # Previous 1d period's high (but we need 1w)
                pl = l_1w_aligned[prev_1w_idx]  # Previous 1d period's low
                pc = c_1w_aligned[prev_1w_idx]  # Previous 1d period's close
                
                # To get true 1w bar data, we need to look back further
                # Since we aligned 1w data to 1d, each 1w value repeats for 5 bars
                # So we can use the aligned arrays directly with proper indexing
                # The aligned 1w data is already stretched to 1d resolution
                # We need to ensure we're using completed 1w bar data
                
                # For 1d timeframe, 1w data updates every 5 bars
                # So we look back to the previous multiple of 5 to get completed 1w bar
                lookback_idx = ((i // 5) * 5) - 5  # Previous completed 1w bar
                
                if lookback_idx >= 0:
                    ph = h_1w_aligned[lookback_idx]
                    pl = l_1w_aligned[lookback_idx]
                    pc = c_1w_aligned[lookback_idx]
                    
                    # Calculate Camarilla levels
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
                                prices['close'].iloc[i] > ema50_1w_aligned[i]):
                                position = 1
                                signals[i] = 0.25
                            # Short breakdown: price < Camarilla L3 with volume spike AND 1w downtrend
                            elif (prices['close'].iloc[i] < camarilla_l3 and 
                                  vol_spike.iloc[i] and 
                                  prices['close'].iloc[i] < ema50_1w_aligned[i]):
                                position = -1
                                signals[i] = -0.25
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