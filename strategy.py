#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R (14) with 1d EMA (34) Trend Filter
# Long when Williams %R crosses above -80 (oversold reversal) and price > 1d EMA34
# Short when Williams %R crosses below -20 (overbought reversal) and price < 1d EMA34
# Exit when Williams %R crosses -50 (mean reversion)
# Williams %R identifies reversals in overbought/oversold conditions
# EMA34 filter ensures we trade with the daily trend
# Target: 15-30 trades/year by requiring both Williams %R signal and trend alignment

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R (14) on 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if data not ready
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        price = prices['close'].iloc[i]
        wr = williams_r[i]
        wr_prev = williams_r[i-1]
        ema_trend = ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold reversal) with uptrend
            if wr_prev <= -80 and wr > -80 and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought reversal) with downtrend
            elif wr_prev >= -20 and wr < -20 and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when Williams %R crosses -50 (mean reversion)
            exit_signal = False
            
            if position == 1:  # long position
                if wr_prev > -50 and wr <= -50:
                    exit_signal = True
            
            elif position == -1:  # short position
                if wr_prev < -50 and wr >= -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsR14_1dEMA34_Trend_MeanRev"
timeframe = "12h"
leverage = 1.0