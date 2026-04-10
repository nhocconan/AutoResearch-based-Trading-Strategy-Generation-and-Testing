#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 level with volume > 1.3x average AND daily close > daily EMA50
# - Short when price breaks below Camarilla L3 level with volume > 1.3x average AND daily close < daily EMA50
# - Exit when price retreats to Camarilla H4/L4 levels or volume drops below average
# - Daily trend filter ensures alignment with major trend
# - Volume confirmation prevents false breakouts
# - Targets 12-25 trades/year (50-100 total over 4 years) to avoid fee drag
# - Camarilla pivots work well in ranging markets; combined with trend/volume filters for breakouts

name = "12h_1d_camarilla_breakout_volume_trend_v1"
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
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute volume confirmation: > 1.3x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.3 * volume_20_avg)
    
    # Pre-compute volume filter: < average volume for exit
    vol_normal = prices['volume'] < volume_20_avg
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_20_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Calculate Camarilla levels from previous 1d bar
        # Need previous day's OHLC (from 1d data aligned to current 12h bar)
        h_1d = df_1d['high'].values
        l_1d = df_1d['low'].values
        c_1d = df_1d['close'].values
        
        # Get the completed 1d bar index for current 12h bar
        # Since 12h bars are 2 per day, we need the previous completed 1d bar
        # align_htf_to_ltf already handles the delay, so we use the previous value
        if i >= 2:  # Need at least 2 bars to have previous day
            prev_day_idx = i // 2 - 1  # Approximate, but we'll use aligned arrays properly
            # Better approach: use the aligned 1d data to get previous day's values
            # We'll use the aligned arrays shifted by 1 to get previous completed day
            pass
        
        # Simpler: calculate Camarilla levels using rolling window on 1d data aligned to 12h
        # We need the previous completed 1d bar's OHLC
        # Create aligned arrays for 1d OHLC
        h_1d_aligned = align_htf_to_ltf(prices, df_1d, h_1d)
        l_1d_aligned = align_htf_to_ltf(prices, df_1d, l_1d)
        c_1d_aligned = align_htf_to_ltf(prices, df_1d, c_1d)
        
        # Get previous completed 1d bar values (shifted by 1 to avoid look-ahead)
        if i >= 1:
            ph = h_1d_aligned[i-1]  # Previous day's high
            pl = l_1d_aligned[i-1]  # Previous day's low
            pc = c_1d_aligned[i-1]  # Previous day's close
            
            if not (np.isnan(ph) or np.isnan(pl) or np.isnan(pc)):
                # Calculate Camarilla levels
                range_val = ph - pl
                if range_val > 0:
                    camarilla_h3 = pc + (range_val * 1.1 / 4)
                    camarilla_l3 = pc - (range_val * 1.1 / 4)
                    camarilla_h4 = pc + (range_val * 1.1 / 2)
                    camarilla_l4 = pc - (range_val * 1.1 / 2)
                    
                    if position == 0:  # Flat - look for new breakout entries
                        # Long breakout: price > Camarilla H3 with volume spike AND daily uptrend
                        if (prices['close'].iloc[i] > camarilla_h3 and 
                            vol_spike.iloc[i] and 
                            prices['close'].iloc[i] > ema50_1d_aligned[i]):
                            position = 1
                            signals[i] = 0.25
                        # Short breakdown: price < Camarilla L3 with volume spike AND daily downtrend
                        elif (prices['close'].iloc[i] < camarilla_l3 and 
                              vol_spike.iloc[i] and 
                              prices['close'].iloc[i] < ema50_1d_aligned[i]):
                            position = -1
                            signals[i] = -0.25
                    else:  # Have position - look for exit
                        # Exit conditions:
                        # 1. Price retreats to Camarilla H4/L4 levels
                        # 2. Volume drops below average (loss of momentum)
                        if position == 1:  # Long position
                            if (prices['close'].iloc[i] < camarilla_h4 or 
                                vol_normal.iloc[i]):
                                position = 0
                                signals[i] = 0.0
                            else:
                                signals[i] = 0.25  # Hold long
                        elif position == -1:  # Short position
                            if (prices['close'].iloc[i] > camarilla_l4 or 
                                vol_normal.iloc[i]):
                                position = 0
                                signals[i] = 0.0
                            else:
                                signals[i] = -0.25  # Hold short
                else:
                    signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            else:
                signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
        else:
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
    
    return signals