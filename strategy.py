#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d 50-period EMA crossover with 1w EMA200 trend filter and volume confirmation.
# Long when EMA(50) crosses above EMA(200) on daily and weekly EMA200 confirms uptrend.
# Short when EMA(50) crosses below EMA(200) on daily and weekly EMA200 confirms downtrend.
# Volume > 1.3x 20-period average confirms momentum. Designed for low turnover (10-25 trades/year).
# Works in bull/bear: weekly trend filter prevents counter-trend trades in strong regimes.

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d EMA(50) and EMA(200)
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 1w EMA(200) for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if np.isnan(ema_50[i]) or np.isnan(ema_200[i]) or np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirm = volume > 1.3 * vol_ma[i]
        
        # Daily EMA crossover signals
        ema50_above_ema200 = ema_50[i] > ema_200[i]
        ema50_above_ema200_prev = ema_50[i-1] > ema_200[i-1]
        
        bullish_cross = ema50_above_ema200 and not ema50_above_ema200_prev
        bearish_cross = not ema50_above_ema200 and ema50_above_ema200_prev
        
        # Weekly trend filter
        weekly_uptrend = ema_200_1w_aligned[i] > ema_200_1w_aligned[i-1]  # rising weekly EMA200
        weekly_downtrend = ema_200_1w_aligned[i] < ema_200_1w_aligned[i-1]  # falling weekly EMA200
        
        if position == 0:
            if volume_confirm:
                # Long: bullish crossover + weekly uptrend
                if bullish_cross and weekly_uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: bearish crossover + weekly downtrend
                elif bearish_cross and weekly_downtrend:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions: opposite crossover OR weekly trend reversal
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on bearish crossover or weekly trend turns down
                if bearish_cross or not weekly_uptrend:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on bullish crossover or weekly trend turns up
                if bullish_cross or not weekly_downtrend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_EMA50_200_Crossover_1wEMA200_Trend_Volume"
timeframe = "1d"
leverage = 1.0