#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1-week trend filter and volume confirmation
# Long when Williams %R crosses above -80 (oversold reversal) in uptrend (close > 1w EMA50) with volume spike
# Short when Williams %R crosses below -20 (overbought reversal) in downtrend (close < 1w EMA50) with volume spike
# Exit when Williams %R returns to -50 (mean reversion) or trend reverses
# Williams %R is effective at catching reversals in ranging markets, which dominates 2025+ BTC/ETH
# Trend filter ensures we trade with the higher timeframe momentum
# Volume confirmation filters out false reversals
# Designed for low trade frequency (~15-30/year) to minimize fee drain.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 50-period EMA on 1w close for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA to 6h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams %R on 6h data (14-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema_val = ema_50_1w_aligned[i]
        wr = williams_r[i]
        
        # Previous Williams %R for crossover detection
        wr_prev = williams_r[i-1]
        
        # Volume filter: current volume > 1.8 * 20-period average
        vol_spike = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long conditions: Williams %R crosses above -80 + uptrend + volume spike
            if wr > -80 and wr_prev <= -80 and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R crosses below -20 + downtrend + volume spike
            elif wr < -20 and wr_prev >= -20 and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: Williams %R returns to -50 or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Williams %R returns to -50 or trend turns down
                if wr >= -50 or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Williams %R returns to -50 or trend turns up
                if wr <= -50 or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR_1wEMA50_Volume"
timeframe = "6h"
leverage = 1.0