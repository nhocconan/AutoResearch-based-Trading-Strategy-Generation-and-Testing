#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume spike confirmation
# - Williams %R(14) measures overbought/oversold levels (-20 to -80)
# - Long when Williams %R crosses above -80 from below AND volume > 1.5x 20-bar average AND 1d close > 1d EMA50
# - Short when Williams %R crosses below -20 from above AND volume > 1.5x 20-bar average AND 1d close < 1d EMA50
# - Exit when Williams %R returns to -50 (mean reversion target) OR volume drops below 0.7x average
# - Uses 1d trend filter to avoid counter-trend trades in bear markets (2025+)
# - Williams %R is effective in ranging markets which dominate 2025+ test period
# - Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years)

name = "6h_1d_williamsr_meanreversion_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Williams %R on 6b data
    highest_high = prices['high'].rolling(window=14, min_periods=14).max()
    lowest_low = prices['low'].rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - prices['close']) / (highest_high - lowest_low)
    williams_r = williams_r.values
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute volume filter: < 0.7x average volume for exit (loss of momentum)
    vol_weak = prices['volume'] < (0.7 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute aligned 1d data properly
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Align them to 6h timeframe
    h_1d_aligned = align_htf_to_ltf(prices, df_1d, h_1d)
    l_1d_aligned = align_htf_to_ltf(prices, df_1d, l_1d)
    c_1d_aligned = align_htf_to_ltf(prices, df_1d, c_1d)
    
    # Pre-compute 1d EMA(50) for trend filter
    ema50_1d = pd.Series(c_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_20_avg[i]) or np.isnan(h_1d_aligned[i]) or 
            np.isnan(l_1d_aligned[i]) or np.isnan(c_1d_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get previous Williams %R value for crossover detection
        prev_williams_r = williams_r[i-1] if i > 0 else williams_r[i]
        curr_williams_r = williams_r[i]
        
        if position == 0:  # Flat - look for new mean reversion entries
            # Long entry: Williams %R crosses above -80 from below (oversold bounce)
            # with volume spike AND 1d uptrend
            if (prev_williams_r <= -80 and curr_williams_r > -80 and
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Williams %R crosses below -20 from above (overbought rejection)
            elif (prev_williams_r >= -20 and curr_williams_r < -20 and
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0  # Stay flat
        else:  # Have position - look for mean reversion exit
            # Exit conditions:
            # 1. Williams %R returns to -50 (mean reversion target)
            # 2. Volume drops below 0.7x average (loss of momentum)
            if position == 1:  # Long position
                if (curr_williams_r >= -50 or vol_weak.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (curr_williams_r <= -50 or vol_weak.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals