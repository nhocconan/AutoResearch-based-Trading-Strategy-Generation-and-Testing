#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla H4/L4 breakout with 1d trend filter and volume confirmation
# - Long when price breaks above H4 level AND 1d EMA200 rising AND volume > 1.8x 20-bar avg
# - Short when price breaks below L4 level AND 1d EMA200 falling AND volume > 1.8x 20-bar avg
# - Exit when price crosses the Pivot level (mean reversion to equilibrium)
# - Uses 1d EMA200 for strong trend filter to avoid weak/choppy markets
# - H4/L4 levels provide tighter breakouts than H3/L3 for higher quality signals
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 20-35 trades/year on 4h timeframe (80-140 total over 4 years)
# - Focus on BTC/ETH as primary symbols, SOL as secondary

name = "4h_1d_camarilla_h4l4_breakout_volume_trend_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Camarilla levels from daily data (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels calculation
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # H4, L4 levels (tighter breakout levels)
    h4 = close_1d + (range_1d * 1.1 / 2)
    l4 = close_1d - (range_1d * 1.1 / 2)
    # Pivot level for exit
    piv = pivot
    
    # Align HTF levels to LTF (4h)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    piv_aligned = align_htf_to_ltf(prices, df_1d, piv)
    
    # Pre-compute 1d EMA(200) for strong trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Pre-compute volume confirmation: > 1.8x 20-period average (slightly looser for more signals)
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.8 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(piv_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above H4 AND 1d uptrend (price > EMA200) with volume spike
            if (prices['close'].iloc[i] > h4_aligned[i] and 
                prices['close'].iloc[i] > ema200_1d_aligned[i] and  # price above 1d EMA200 = uptrend
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below L4 AND 1d downtrend (price < EMA200) with volume spike
            elif (prices['close'].iloc[i] < l4_aligned[i] and 
                  prices['close'].iloc[i] < ema200_1d_aligned[i] and  # price below 1d EMA200 = downtrend
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to pivot (mean reversion)
            # Exit when price crosses the pivot level
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