#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Long when price breaks above H3 level AND 1w close > 1w open (bullish weekly candle) AND volume > 2.0x 20-bar avg
# - Short when price breaks below L3 level AND 1w close < 1w open (bearish weekly candle) AND volume > 2.0x 20-bar avg
# - Exit when price returns to Pivot level (mean reversion to equilibrium)
# - Uses 1w candle direction for trend filter to avoid counter-trend trades and capture major moves
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Camarilla pivots work well in ranging markets; weekly trend filter adds directional bias in trends

name = "12h_1w_camarilla_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute Camarilla pivot levels from daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # H3, L3 levels (most important for breakouts)
    h3 = close_1d + (range_1d * 1.1 / 4)
    l3 = close_1d - (range_1d * 1.1 / 4)
    # Pivot level for exit
    piv = pivot
    
    # Align HTF levels to LTF
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    piv_aligned = align_htf_to_ltf(prices, df_1d, piv)
    
    # Pre-compute 1w trend filter: bullish if close > open, bearish if close < open
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    weekly_bullish = close_1w > open_1w
    weekly_bearish = close_1w < open_1w
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # Pre-compute volume confirmation: > 2.0x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (2.0 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(piv_aligned[i]) or np.isnan(weekly_bullish_aligned[i]) or 
            np.isnan(weekly_bearish_aligned[i]) or np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above H3 AND weekly bullish AND volume spike
            if (prices['close'].iloc[i] > h3_aligned[i] and 
                weekly_bullish_aligned[i] > 0.5 and 
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below L3 AND weekly bearish AND volume spike
            elif (prices['close'].iloc[i] < l3_aligned[i] and 
                  weekly_bearish_aligned[i] > 0.5 and 
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to pivot (mean reversion)
            # Exit when price returns to pivot level
            exit_signal = False
            if position == 1:  # Long position
                if prices['close'].iloc[i] <= piv_aligned[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if prices['close'].iloc[i] >= piv_aligned[i]:
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