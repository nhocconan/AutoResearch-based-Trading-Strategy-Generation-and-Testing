#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1w trend filter and volume spike confirmation
# - Williams %R(14): overbought > -20, oversold < -80
# - 1w EMA200 as trend filter: long only when price > EMA200, short only when price < EMA200
# - Volume confirmation: current volume > 2.0x 24-period average (captures panic/surge)
# - Entry: Williams %R crosses below -80 (oversold) in uptrend OR crosses above -20 (overbought) in downtrend
# - Exit: Williams %R crosses above -50 (for longs) or below -50 (for shorts)
# - Discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought rallies in downtrend)

name = "6h_1w_williamsr_meanrev_volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Pre-compute Williams %R on 6h timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    
    # Pre-compute 6h volume confirmation (24-period average)
    volume_sma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema200_1w_aligned[i]) or
            np.isnan(volume_sma_24[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price = close[i]
        volume_current = volume[i]
        wr = williams_r[i]
        
        # Trend filter from 1w EMA200
        uptrend = price > ema200_1w_aligned[i]
        downtrend = price < ema200_1w_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 24-period average
        vol_confirm = volume_current > 2.0 * volume_sma_24[i]
        
        # Williams %R levels
        oversold = wr < -80
        overbought = wr > -20
        exit_level = wr > -50  # for longs
        exit_level_short = wr < -50  # for shorts
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Williams %R crosses below -80 (oversold) in uptrend + volume confirmation
        if i > 100:
            wr_prev = williams_r[i-1]
            crossed_oversold = (wr_prev >= -80) and (wr < -80)
            if crossed_oversold and uptrend and vol_confirm:
                enter_long = True
        
        # Short: Williams %R crosses above -20 (overbought) in downtrend + volume confirmation
        if i > 100:
            wr_prev = williams_r[i-1]
            crossed_overbought = (wr_prev <= -20) and (wr > -20)
            if crossed_overbought and downtrend and vol_confirm:
                enter_short = True
        
        # Exit conditions: Williams %R crosses back through -50
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Williams %R crosses above -50
            if i > 100:
                wr_prev = williams_r[i-1]
                crossed_exit = (wr_prev <= -50) and (wr > -50)
                exit_long = crossed_exit
        elif position == -1:
            # Exit short if Williams %R crosses below -50
            if i > 100:
                wr_prev = williams_r[i-1]
                crossed_exit_short = (wr_prev >= -50) and (wr < -50)
                exit_short = crossed_exit_short
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals