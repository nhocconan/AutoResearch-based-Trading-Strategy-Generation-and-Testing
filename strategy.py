#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 12h EMA trend filter and volume spike
# Long when Williams %R crosses above -50 (bullish momentum) + close > 12h EMA50 (uptrend) + volume spike
# Short when Williams %R crosses below -50 (bearish momentum) + close < 12h EMA50 (downtrend) + volume spike
# Exit when Williams %R crosses -30/-70 or trend reverses
# Williams %R is a momentum oscillator that measures overbought/oversold levels
# Designed for low trade frequency (~20-40/year) to minimize fee drain.
# Works in bull/bear by combining momentum with trend following and volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 50-period EMA on 12h close for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Williams %R on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 14-period Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    hh_ll = highest_high - lowest_low
    willr = np.where(hh_ll != 0, (highest_high - close) / hh_ll * -100, -50.0)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    prev_willr = -50.0  # previous Williams %R value for crossover detection
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(willr[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            prev_willr = willr[i] if not np.isnan(willr[i]) else prev_willr
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        willr_val = willr[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        # Williams %R crossovers
        willr_cross_above_50 = prev_willr <= -50.0 and willr_val > -50.0
        willr_cross_below_50 = prev_willr >= -50.0 and willr_val < -50.0
        willr_cross_above_30 = prev_willr <= -30.0 and willr_val > -30.0
        willr_cross_below_70 = prev_willr >= -70.0 and willr_val < -70.0
        
        if position == 0:
            # Long conditions: Williams %R crosses above -50 + uptrend + volume spike
            if willr_cross_above_50 and price > ema_50_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R crosses below -50 + downtrend + volume spike
            elif willr_cross_below_50 and price < ema_50_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: Williams %R crosses -30/-70 or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Williams %R crosses above -30 (overbought) or trend turns down
                if willr_cross_above_30 or price < ema_50_aligned[i]:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Williams %R crosses below -70 (oversold) or trend turns up
                if willr_cross_below_70 or price > ema_50_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
        
        prev_willr = willr_val
    
    return signals

name = "4h_WilliamsR_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0