#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1w Williams %R extreme readings with 1d trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions on weekly timeframe
# Extreme readings (>80 for oversold, <20 for overbought) combined with 1d EMA trend direction
# Volume confirmation ensures breakout validity
# Discrete sizing 0.25 to target ~20-50 trades/year and minimize fee drag
# Works in bull/bear markets: mean reversion at extremes in ranging markets, trend continuation in trending markets

name = "4h_1w_williamsr_1d_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 10:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1w Williams %R(14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close_1w) / (highest_high - lowest_low)) * -100,
        -50  # neutral when no range
    )
    
    # Align 1w Williams %R to 4h timeframe with 1-bar delay (wait for weekly close)
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    
    # 4h volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        williams_r_val = williams_r_aligned[i]
        ema_trend = close[i] > ema_50_1d_aligned[i]  # Price above 1d EMA50 = uptrend
        vol_conf = volume_confirmed[i] if not np.isnan(vol_ma_20[i]) else False
        
        if position == 1:  # Long position
            # Exit long if Williams %R returns from oversold or trend turns down
            if williams_r_val > -80 or not ema_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if Williams %R returns from overbought or trend turns up
            if williams_r_val < -20 or ema_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long when Williams %R shows extreme oversold (<-80) in uptrend with volume
            if williams_r_val < -80 and ema_trend and vol_conf:
                position = 1
                signals[i] = 0.25
            # Enter short when Williams %R shows extreme overbought (>-20) in downtrend with volume
            elif williams_r_val > -20 and not ema_trend and vol_conf:
                position = -1
                signals[i] = -0.25
    
    return signals