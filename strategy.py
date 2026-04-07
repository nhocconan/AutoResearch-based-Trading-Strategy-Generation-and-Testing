#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Exponential Moving Average Crossover with Daily Trend Filter and Volume Confirmation
# Long when EMA(20) > EMA(50), price > EMA(50), and daily close > weekly close (bullish bias)
# Short when EMA(20) < EMA(50), price < EMA(50), and daily close < weekly close (bearish bias)
# Volume must be > 1.2x 20-period average for confirmation
# Exit when EMA crossover reverses or volume dries up
# Position size: 0.25 (25% of capital)
# Uses EMA crossover for momentum, higher timeframe for trend bias, volume for confirmation
# Target: 100-200 total trades over 4 years (25-50/year)

name = "6h_ema_crossover_daily_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA calculations
    ema20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Volume average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    close_daily = df_daily['close'].values
    # Weekly data for stronger trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        # Fallback to daily if weekly not available
        weekly_trend = np.zeros(len(prices))
    else:
        close_weekly = df_weekly['close'].values
        weekly_trend = align_htf_to_ltf(prices, df_weekly, close_weekly)
    
    # Align daily close to 6h timeframe
    daily_close_aligned = align_htf_to_ltf(prices, df_daily, close_daily)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(ema20[i]) or np.isnan(ema50[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(daily_close_aligned[i])):
            if len(df_weekly) >= 2 and np.isnan(weekly_trend[i]):
                if position != 0:
                    signals[i] = position * 0.25
                else:
                    signals[i] = 0.0
                continue
            elif len(df_weekly) < 2:
                # No weekly data, continue with available data
                pass
            else:
                if position != 0:
                    signals[i] = position * 0.25
                else:
                    signals[i] = 0.0
                continue
        
        # Volume confirmation: volume > 1.2x average
        vol_confirm = volume[i] > 1.2 * volume_ma[i]
        
        # Trend filter: daily close > weekly close for bullish bias, < for bearish
        if len(df_weekly) >= 2:
            bullish_bias = daily_close_aligned[i] > weekly_trend[i]
            bearish_bias = daily_close_aligned[i] < weekly_trend[i]
        else:
            # Fallback: use daily close vs its own 10-period MA for trend
            daily_ma10 = pd.Series(close_daily).rolling(window=10, min_periods=10).mean().values
            daily_ma10_aligned = align_htf_to_ltf(prices, df_daily, daily_ma10)
            bullish_bias = daily_close_aligned[i] > daily_ma10_aligned[i]
            bearish_bias = daily_close_aligned[i] < daily_ma10_aligned[i]
        
        if position == 1:  # long position
            # Exit: EMA crossover reverses or volume dries up
            if ema20[i] <= ema50[i] or not vol_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: EMA crossover reverses or volume dries up
            if ema20[i] >= ema50[i] or not vol_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with EMA crossover, volume confirmation, and trend bias
            # Long: EMA(20) > EMA(50), price > EMA(50), bullish bias, volume confirmation
            if (ema20[i] > ema50[i] and
                close[i] > ema50[i] and
                bullish_bias and
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short: EMA(20) < EMA(50), price < EMA(50), bearish bias, volume confirmation
            elif (ema20[i] < ema50[i] and
                  close[i] < ema50[i] and
                  bearish_bias and
                  vol_confirm):
                signals[i] = -0.25
                position = -1
    
    return signals