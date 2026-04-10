#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 level with volume > 1.3x average AND daily close > daily EMA20
# - Short when price breaks below Camarilla L3 level with volume > 1.3x average AND daily close < daily EMA20
# - Exit when price retreats to Camarilla H4/L4 levels
# - Daily trend filter ensures alignment with intermediate trend
# - Volume confirmation prevents false breakouts
# - Targets 12-30 trades/year (50-120 total over 4 years) to avoid fee drag

name = "12h_1d_camarilla_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d EMA(20) for trend filter
    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Pre-compute volume confirmation: > 1.3x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.3 * volume_20_avg)
    
    # Pre-compute 1d OHLC aligned to 12h for Camarilla calculation
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    h_1d_aligned = align_htf_to_ltf(prices, df_1d, h_1d)
    l_1d_aligned = align_htf_to_ltf(prices, df_1d, l_1d)
    c_1d_aligned = align_htf_to_ltf(prices, df_1d, c_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema20_1d_aligned[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(h_1d_aligned[i]) or np.isnan(l_1d_aligned[i]) or np.isnan(c_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
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
                            prices['close'].iloc[i] > ema20_1d_aligned[i]):
                            position = 1
                            signals[i] = 0.25
                        # Short breakdown: price < Camarilla L3 with volume spike AND daily downtrend
                        elif (prices['close'].iloc[i] < camarilla_l3 and 
                              vol_spike.iloc[i] and 
                              prices['close'].iloc[i] < ema20_1d_aligned[i]):
                            position = -1
                            signals[i] = -0.25
                    else:  # Have position - look for exit
                        # Exit when price retreats to Camarilla H4/L4 levels
                        if position == 1:  # Long position
                            if prices['close'].iloc[i] < camarilla_h4:
                                position = 0
                                signals[i] = 0.0
                            else:
                                signals[i] = 0.25  # Hold long
                        elif position == -1:  # Short position
                            if prices['close'].iloc[i] > camarilla_l4:
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