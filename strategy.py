#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly EMA34 trend filter and daily Williams %R extreme reversals
# Long when: price > weekly EMA34 AND daily Williams %R < -80 (oversold) AND 6h close > 6h open (bullish candle)
# Short when: price < weekly EMA34 AND daily Williams %R > -20 (overbought) AND 6h close < 6h open (bearish candle)
# Exit when: price crosses weekly EMA34 in opposite direction OR Williams %R returns to neutral range (-50)
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Weekly EMA34 provides robust trend filter avoiding whipsaws in sideways markets
# Daily Williams %R extremes capture short-term exhaustion points with high reversal probability
# 6h candle direction ensures alignment with immediate momentum
# Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought rallies in downtrend)

name = "6h_WeeklyEMA34_DailyWilliamsR_Extreme"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data ONCE before loop for EMA34
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Get daily data ONCE before loop for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need sufficient data for Williams %R
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly EMA34
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate daily Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Williams %R = -100 * (HHV - Close) / (HHV - LLV) where HHV/LLV are 14-period
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    # Handle division by zero when HHV == LLV
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r, additional_delay_bars=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(williams_r_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above weekly EMA34 AND daily Williams %R oversold (< -80) AND bullish 6h candle
            if (close[i] > ema_34_1w_aligned[i] and 
                williams_r_aligned[i] < -80 and 
                close[i] > open_price[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly EMA34 AND daily Williams %R overbought (> -20) AND bearish 6h candle
            elif (close[i] < ema_34_1w_aligned[i] and 
                  williams_r_aligned[i] > -20 and 
                  close[i] < open_price[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below weekly EMA34 OR Williams %R returns to neutral (> -50)
            if close[i] <= ema_34_1w_aligned[i] or williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above weekly EMA34 OR Williams %R returns to neutral (< -50)
            if close[i] >= ema_34_1w_aligned[i] or williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals