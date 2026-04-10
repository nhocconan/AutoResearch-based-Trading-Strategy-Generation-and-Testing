#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with weekly trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 level with volume > 1.8x average AND weekly close > weekly EMA34
# - Short when price breaks below Camarilla L3 level with volume > 1.8x average AND weekly close < weekly EMA34
# - Exit when price retests Camarilla H4/L4 levels or volume drops below average
# - Weekly trend filter ensures alignment with major trend
# - Volume confirmation prevents false breakouts
# - Camarilla pivots work well in ranging markets with clear intraday structure
# - Targets 12-30 trades/year (50-120 total over 4 years) to avoid fee drag

name = "12h_1w_camarilla_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Pre-compute 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Pre-compute volume confirmation: > 1.8x 24-period average (2 days of 12h bars)
    volume_24_avg = prices['volume'].rolling(window=24, min_periods=24).mean().values
    vol_spike = prices['volume'] > (1.8 * volume_24_avg)
    
    # Pre-compute volume filter: < average volume for exit
    vol_normal = prices['volume'] < volume_24_avg
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Need at least 24h of data to calculate Camarilla (2 previous 12h bars)
        if i < 2:
            signals[i] = 0.0
            continue
            
        # Calculate Camarilla pivot levels using previous 12h bar (HLC of i-1)
        # Camarilla formula: based on previous period's range
        high_prev = prices['high'].iloc[i-1]
        low_prev = prices['low'].iloc[i-1]
        close_prev = prices['close'].iloc[i-1]
        
        pivot = (high_prev + low_prev + close_prev) / 3.0
        range_val = high_prev - low_prev
        
        # Camarilla levels
        h3 = pivot + (range_val * 1.1 / 4)  # ~1.1 * range / 4
        l3 = pivot - (range_val * 1.1 / 4)
        h4 = pivot + (range_val * 1.1 / 2)  # ~1.1 * range / 2
        l4 = pivot - (range_val * 1.1 / 2)
        
        # Skip if any required data is invalid
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_24_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long breakout: price > H3 with volume spike AND weekly uptrend
            if (prices['close'].iloc[i] > h3 and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema34_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown: price < L3 with volume spike AND weekly downtrend
            elif (prices['close'].iloc[i] < l3 and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema34_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price retests H4/L4 levels (strong reversal signal)
            # 2. Volume drops below average (loss of momentum)
            if position == 1:  # Long position
                if (prices['close'].iloc[i] < h4 or 
                    vol_normal.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (prices['close'].iloc[i] > l4 or 
                    vol_normal.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals