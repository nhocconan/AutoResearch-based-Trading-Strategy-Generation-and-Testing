#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d trend filter and volume confirmation
# - Long when price breaks above H3 level with volume > 1.5x average AND daily close > daily EMA20
# - Short when price breaks below L3 level with volume > 1.5x average AND daily close < daily EMA20
# - Exit when price retests H4/L4 levels or volume drops below average
# - Daily trend filter ensures alignment with intermediate trend
# - Volume confirmation prevents false breakouts
# - Targets 12-37 trades/year (50-150 total over 4 years) to avoid fee drag
# - Camarilla pivots work well in both trending and ranging markets when combined with trend and volume filters

name = "12h_1d_camarilla_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute Camarilla pivot levels from previous 1d bar (standard calculation)
    # H4 = Close + 1.5*(High-Low), H3 = Close + 1.0*(High-Low), etc.
    # L4 = Close - 1.5*(High-Low), L3 = Close - 1.0*(High-Low), etc.
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels for each 1d bar
    camarilla_h4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_h3 = close_1d + 1.0 * (high_1d - low_1d)
    camarilla_l3 = close_1d - 1.0 * (high_1d - low_1d)
    camarilla_l4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align HTF pivot levels to 12h timeframe (wait for completed 1d bar)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Pre-compute 1d EMA(20) for trend filter
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average on 12h timeframe
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute volume filter: < average volume for exit
    vol_normal = prices['volume'] < volume_20_avg
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or 
            np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or
            np.isnan(ema20_1d_aligned[i]) or np.isnan(volume_20_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long breakout: price > H3 level with volume spike AND daily uptrend
            if (prices['high'].iloc[i] > h3_1d_aligned[i] and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema20_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown: price < L3 level with volume spike AND daily downtrend
            elif (prices['low'].iloc[i] < l3_1d_aligned[i] and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema20_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price retests H4/L4 levels (strong reversal signal)
            # 2. Volume drops below average (loss of momentum)
            if position == 1:  # Long position
                if (prices['low'].iloc[i] < h4_1d_aligned[i] or 
                    vol_normal.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (prices['high'].iloc[i] > l4_1d_aligned[i] or 
                    vol_normal.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals