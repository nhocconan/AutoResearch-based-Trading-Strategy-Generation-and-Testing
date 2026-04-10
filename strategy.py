#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Long when Williams %R(14) < -80 (oversold) AND price > 1d EMA200 AND volume > 1.5x 20-bar average
# - Short when Williams %R(14) > -20 (overbought) AND price < 1d EMA200 AND volume > 1.5x 20-bar average
# - Exit when Williams %R returns to -50 level OR volume drops below 0.7x average
# - Uses 1d EMA200 for strong trend filter to avoid counter-trend trades
# - Moderate volume threshold (1.5x) and Williams %R extremes target 15-25 trades/year
# - Works in both bull (buy oversold dips) and bear (sell overbought rallies) markets

name = "12h_1d_williamsr_meanreversion_volume_trend_v1"
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
    
    # Pre-compute Williams %R(14) on 12h data
    highest_high = prices['high'].rolling(window=14, min_periods=14).max()
    lowest_low = prices['low'].rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - prices['close']) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).fillna(-50).values  # Neutral when range=0
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute volume filter: < 0.7x average volume for exit (loss of momentum)
    vol_weak = prices['volume'] < (0.7 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute aligned 1d data properly
    c_1d = df_1d['close'].values
    
    # Align 1d close to 12h timeframe
    c_1d_aligned = align_htf_to_ltf(prices, df_1d, c_1d)
    
    # Pre-compute 1d EMA(200) for trend filter
    ema200_1d = pd.Series(c_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(williams_r[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new mean reversion entries
            # Long oversold: Williams %R < -80 AND price > 1d EMA200 AND volume spike
            if (williams_r[i] < -80 and 
                prices['close'].iloc[i] > ema200_1d_aligned[i] and 
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short overbought: Williams %R > -20 AND price < 1d EMA200 AND volume spike
            elif (williams_r[i] > -20 and 
                  prices['close'].iloc[i] < ema200_1d_aligned[i] and 
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0  # Stay flat
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Williams %R returns to -50 level (mean reversion complete)
            # 2. Volume drops below 0.7x average (loss of momentum)
            if position == 1:  # Long position
                if (williams_r[i] >= -50 or vol_weak.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (williams_r[i] <= -50 or vol_weak.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals