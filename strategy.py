#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R + 1d EMA trend + volume confirmation.
# Williams %R measures momentum overbought/oversold levels.
# Enter long when Williams %R crosses above -80 (oversold) in 1d uptrend with volume expansion.
# Enter short when Williams %R crosses below -20 (overbought) in 1d downtrend with volume expansion.
# Uses Williams %R(14) for momentum and EMA(50) for 1d trend filter.
# Designed for 20-40 trades/year on 4h timeframe with focus on mean reversion in trending markets.
# Volume filter ensures reversals have conviction, reducing false signals.
# 1d trend filter prevents counter-trend trading in choppy markets.

name = "4h_1d_williams_r_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate volume moving average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):  # Start after Williams %R period
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(williams_r[i-1]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5 * 20-period average volume
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Determine 1d trend direction
        is_uptrend = close[i] > ema_50_1d_aligned[i]
        is_downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Williams %R conditions
        wr_above_oversold = williams_r[i] > -80
        wr_below_oversold_prev = williams_r[i-1] <= -80
        wr_below_overbought = williams_r[i] < -20
        wr_above_overbought_prev = williams_r[i-1] >= -20
        
        # Entry conditions
        bullish_entry = wr_above_oversold and wr_below_oversold_prev and vol_filter and is_uptrend
        bearish_entry = wr_below_overbought and wr_above_overbought_prev and vol_filter and is_downtrend
        
        # Exit conditions: opposite Williams %R signal
        exit_long = wr_below_overbought and wr_above_overbought_prev
        exit_short = wr_above_oversold and wr_below_oversold_prev
        
        # Priority: entry > exit > hold
        if bullish_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals