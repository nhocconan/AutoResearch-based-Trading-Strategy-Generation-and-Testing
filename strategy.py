#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with volume confirmation and 12h trend filter
# - Long when Williams %R(14) < -80 (oversold) with volume > 1.5x 20-bar average AND 12h close > 12h EMA20
# - Short when Williams %R(14) > -20 (overbought) with volume > 1.5x 20-bar average AND 12h close < 12h EMA20
# - Exit when Williams %R returns to -50 level OR volume drops below 0.7x average
# - Uses 12h trend filter to avoid counter-trend trades and targets 20-35 trades/year (80-140 total over 4 years)
# - Williams %R is effective in ranging markets and captures reversals in both bull/bear regimes
# - Moderate volume threshold reduces whipsaws while maintaining sufficient trade frequency

name = "4h_12h_williamsr_meanreversion_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute Williams %R(14) on 4h data
    highest_high = prices['high'].rolling(window=14, min_periods=14).max()
    lowest_low = prices['low'].rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - prices['close']) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).fillna(50).values  # Neutral when range=0
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute volume filter: < 0.7x average volume for exit (loss of momentum)
    vol_weak = prices['volume'] < (0.7 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute aligned 12h data properly
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    # Align them to 4h timeframe
    h_12h_aligned = align_htf_to_ltf(prices, df_12h, h_12h)
    l_12h_aligned = align_htf_to_ltf(prices, df_12h, l_12h)
    c_12h_aligned = align_htf_to_ltf(prices, df_12h, c_12h)
    
    # Pre-compute 12h EMA(20) for trend filter
    ema20_12h = pd.Series(c_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(ema20_12h_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new mean reversion entries
            # Long entry: oversold + volume spike + 12h uptrend
            if (williams_r[i] < -80 and 
                vol_spike[i] and 
                prices['close'].iloc[i] > ema20_12h_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: overbought + volume spike + 12h downtrend
            elif (williams_r[i] > -20 and 
                  vol_spike[i] and 
                  prices['close'].iloc[i] < ema20_12h_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0  # Stay flat
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Williams %R returns to mean reversion level (-50)
            # 2. Volume drops below 0.7x average (loss of momentum)
            williams_r_mean = -50
            if position == 1:  # Long position
                if (williams_r[i] > williams_r_mean or 
                    vol_weak[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (williams_r[i] < williams_r_mean or 
                    vol_weak[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals