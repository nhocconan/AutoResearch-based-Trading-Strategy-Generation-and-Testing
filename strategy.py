#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 12h trend filter + volume confirmation
# - Williams %R(14) on 6h: oversold < -80, overbought > -20
# - 12h EMA(50) trend filter: only long when price > EMA50, short when price < EMA50
# - Volume confirmation: current volume > 1.8x 20-period average
# - Entry: Williams %R crosses above -80 (long) or below -20 (short) with trend and volume
# - Exit: Williams %R crosses above -50 (long exit) or below -50 (short exit)
# - Works in bull/bear markets by combining mean reversion (Williams %R) with trend filter
# - Target: 12-35 trades/year (50-140 total over 4 years) to stay within fee drag limits

name = "6h_12h_williamsr_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 12h data ONCE before loop for EMA trend (MTF rule compliance)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return signals
    
    # Pre-compute 12h EMA(50)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Pre-compute 6h Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Pre-compute 6h volume confirmation (20-period average)
    volume = prices['volume'].values
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        wr = williams_r[i]
        ema_trend = ema_50_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.8x 20-period average
        vol_confirm = volume_current > 1.8 * volume_sma_20[i]
        
        # Williams %R levels
        wr_oversold = -80.0
        wr_overbought = -20.0
        wr_exit = -50.0
        
        # Williams %R crossover detection (need previous value)
        if i > 0:
            wr_prev = williams_r[i-1]
            wr_cross_above_oversold = wr_prev <= wr_oversold and wr > wr_oversold
            wr_cross_below_overbought = wr_prev >= wr_overbought and wr < wr_overbought
            wr_cross_above_exit = wr_prev <= wr_exit and wr > wr_exit
            wr_cross_below_exit = wr_prev >= wr_exit and wr < wr_exit
        else:
            wr_cross_above_oversold = False
            wr_cross_below_overbought = False
            wr_cross_above_exit = False
            wr_cross_below_exit = False
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Williams %R crosses above oversold (-80) with uptrend and volume
        if wr_cross_above_oversold and close_price > ema_trend and vol_confirm:
            enter_long = True
        
        # Short: Williams %R crosses below overbought (-20) with downtrend and volume
        if wr_cross_below_overbought and close_price < ema_trend and vol_confirm:
            enter_short = True
        
        # Exit conditions: Williams %R crosses midline (-50)
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Williams %R crosses above -50
            exit_long = wr_cross_above_exit
        elif position == -1:
            # Exit short if Williams %R crosses below -50
            exit_short = wr_cross_below_exit
        
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