#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Weekly Trend Filter
# - Elder Ray: Bull Power = High - EMA13(Close), Bear Power = EMA13(Close) - Low
# - Long when Bull Power > 0 AND Bear Power rising (less negative) AND Weekly close > Weekly EMA34
# - Short when Bear Power < 0 AND Bull Power falling (less positive) AND Weekly close < Weekly EMA34
# - Exit when Bull Power <= 0 (long) or Bear Power >= 0 (short)
# - Uses weekly trend filter to avoid counter-trend trades in 2022 crash and 2025 bear
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-25 trades/year on 6h timeframe (50-100 total over 4 years)
# - Works in bull (captures strength) and bear (avoids longs in downtrends)

name = "6h_1w_elder_ray_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # Pre-compute Elder Ray components
    ema13 = pd.Series(prices['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = prices['high'].values - ema13
    bear_power = ema13 - prices['low'].values
    
    # Pre-compute weekly trend filter
    c_1w = df_1w['close'].values
    ema34_1w = pd.Series(c_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, c_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(13, n):
        # Skip if any required data is invalid
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(close_1w_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power positive AND Bear Power rising (improving) AND weekly uptrend
            if (bull_power[i] > 0 and 
                i > 13 and bear_power[i] > bear_power[i-1] and  # Bear Power rising (less negative)
                close_1w_aligned[i] > ema34_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Bear Power negative AND Bull Power falling (weakening) AND weekly downtrend
            elif (bear_power[i] < 0 and 
                  i > 13 and bull_power[i] < bull_power[i-1] and  # Bull Power falling (less positive)
                  close_1w_aligned[i] < ema34_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit long when Bull Power <= 0 (buying pressure gone)
            # Exit short when Bear Power >= 0 (selling pressure gone)
            exit_signal = False
            if position == 1:  # Long position
                if bull_power[i] <= 0:
                    exit_signal = True
            elif position == -1:  # Short position
                if bear_power[i] >= 0:
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