#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with volume confirmation and 1w trend filter
# - Long when price breaks above Camarilla H3 level with volume > 2.0x 20-bar average AND 1w close > 1w EMA34
# - Short when price breaks below Camarilla L3 level with volume > 2.0x 20-bar average AND 1w close < 1w EMA34
# - Exit when price retreats to Camarilla H4/L4 levels OR volume drops below 0.6x average
# - Uses 1w trend filter to avoid counter-trend trades in bear markets (2025+)
# - Higher volume threshold (2.0x) and weaker exit filter (0.6x) reduce trade frequency (target: 8-18 trades/year)
# - Focus on BTC/ETH; SOL-only strategies are low value and will be discarded

name = "1d_1w_camarilla_breakout_volume_trend_v4"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Pre-compute volume confirmation: > 2.0x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (2.0 * volume_20_avg)
    
    # Pre-compute volume filter: < 0.6x average volume for exit (loss of momentum)
    vol_weak = prices['volume'] < (0.6 * volume_20_avg)
    
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
    
    # Pre-compute 1w EMA(34) for trend filter
    ema34_1w = pd.Series(c_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    for i in range(34, n):  # Start after warmup period for EMA34
        # Skip if any required data is invalid
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(h_1w_aligned[i]) or np.isnan(l_1w_aligned[i]) or np.isnan(c_1w_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get previous completed 1d bar values (need to shift by 1 to avoid look-ahead)
        if i >= 1:
            prev_idx = i - 1  # Previous completed 1d bar
            
            if not (np.isnan(h_1w_aligned[prev_idx]) or 
                    np.isnan(l_1w_aligned[prev_idx]) or 
                    np.isnan(c_1w_aligned[prev_idx])):
                ph = h_1w_aligned[prev_idx]  # Previous 1d period's high
                pl = l_1w_aligned[prev_idx]  # Previous 1d period's low
                pc = c_1w_aligned[prev_idx]  # Previous 1d period's close
                
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
                            prices['close'].iloc[i] > ema34_1w_aligned[i]):
                            position = 1
                            signals[i] = 0.25
                        # Short breakdown: price < Camarilla L3 with volume spike AND 1w downtrend
                        elif (prices['close'].iloc[i] < camarilla_l3 and 
                              vol_spike.iloc[i] and 
                              prices['close'].iloc[i] < ema34_1w_aligned[i]):
                            position = -1
                            signals[i] = -0.25
                    else:  # Have position - look for exit
                        # Exit conditions:
                        # 1. Price retreats to Camarilla H4/L4 levels
                        # 2. Volume drops below 0.6x average (loss of momentum)
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