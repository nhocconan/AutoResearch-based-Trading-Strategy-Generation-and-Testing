#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Long when price breaks above H3 level with volume > 1.3x average AND weekly close > weekly EMA20
# - Short when price breaks below L3 level with volume > 1.3x average AND weekly close < weekly EMA20
# - Exit when price retests pivot point (PP) or volume drops below average
# - Weekly trend filter ensures alignment with major trend
# - Volume confirmation prevents false breakouts
# - Targets 12-25 trades/year (50-100 total over 4 years) to avoid fee drag
# - Camarilla pivots work well in ranging markets; breakouts capture strong moves in both bull and bear regimes

name = "12h_1w_camarilla_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute Camarilla pivot levels from daily data (using prior day's OHLC)
    high_1d = prices['high'].rolling(window=2, min_periods=2).max().shift(1).values  # prior day high
    low_1d = prices['low'].rolling(window=2, min_periods=2).min().shift(1).values    # prior day low
    close_1d = prices['close'].rolling(window=2, min_periods=2).mean().shift(1).values  # prior day close
    
    # Calculate pivot point (PP)
    PP = (high_1d + low_1d + close_1d) / 3.0
    # Calculate Camarilla levels
    range_hl = high_1d - low_1d
    H3 = PP + (range_hl * 1.1 / 4)
    L3 = PP - (range_hl * 1.1 / 4)
    H4 = PP + (range_hl * 1.1 / 2)
    L4 = PP - (range_hl * 1.1 / 2)
    
    # Pre-compute 1w EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Pre-compute volume confirmation: > 1.3x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.3 * volume_20_avg)
    
    # Pre-compute volume filter: < average volume for exit
    vol_normal = prices['volume'] < volume_20_avg
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(H3[i]) or np.isnan(L3[i]) or np.isnan(PP[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(volume_20_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long breakout: price > H3 with volume spike AND weekly uptrend
            if (prices['high'].iloc[i] > H3[i] and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema20_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown: price < L3 with volume spike AND weekly downtrend
            elif (prices['low'].iloc[i] < L3[i] and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema20_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price retests pivot point (PP) (mean reversion signal)
            # 2. Volume drops below average (loss of momentum)
            if position == 1:  # Long position
                if (prices['low'].iloc[i] < PP[i] or 
                    vol_normal.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (prices['high'].iloc[i] > PP[i] or 
                    vol_normal.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals