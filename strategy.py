#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d trend filter and volume confirmation
# - Long when price breaks above H3 level AND 1d close > 1d EMA50 AND volume > 1.8x 20-bar avg
# - Short when price breaks below L3 level AND 1d close < 1d EMA50 AND volume > 1.8x 20-bar avg
# - Exit when price crosses the 1d EMA50 (mean reversion to trend)
# - Uses 1d EMA50 for trend filter to avoid counter-trend trades
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-30 trades/year on 12h timeframe (48-120 total over 4 years)
# - Camarilla pivots work well in ranging/bear markets which matches 2025+ test conditions
# - Volume confirmation reduces false breakouts
# - 1d EMA50 filter ensures we trade with the higher timeframe trend

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
    
    # Pre-compute Camarilla pivot levels from 1d data
    # Camarilla levels: based on previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels (using standard multipliers)
    H3 = pivot + (range_1d * 1.1 / 4)  # Resistance level 3
    L3 = pivot - (range_1d * 1.1 / 4)  # Support level 3
    H4 = pivot + (range_1d * 1.1 / 2)  # Resistance level 4
    L4 = pivot - (range_1d * 1.1 / 2)  # Support level 4
    
    # Pre-compute volume confirmation: > 1.8x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.8 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute aligned 1d data
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Pre-compute 1d EMA(50) for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    for i in range(50, n):  # Start after EMA50 warmup
        # Skip if any required data is invalid
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(volume_20_avg[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(close_1d_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above H3 AND in 1d uptrend with volume spike
            if (prices['close'].iloc[i] > H3_aligned[i] and 
                close_1d_aligned[i] > ema50_1d_aligned[i] and 
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below L3 AND in 1d downtrend with volume spike
            elif (prices['close'].iloc[i] < L3_aligned[i] and 
                  close_1d_aligned[i] < ema50_1d_aligned[i] and 
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price crosses the 1d EMA50 (mean reversion to trend)
            exit_signal = False
            if position == 1:  # Long position
                if close_1d_aligned[i] < ema50_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if close_1d_aligned[i] > ema50_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals