#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Weekly Bollinger Breakout with Volume and Trend Filter
# Targets 15-25 trades/year on daily timeframe
# Works in bull markets via breakouts, bear via mean reversion at Bollinger bands
# Uses weekly trend filter to avoid counter-trend trades
# Volume confirmation reduces false breakouts
# Position size 0.25 to control drawdown

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_weekly = df_weekly['close'].values
    ema50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Load daily data for Bollinger Bands - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate Bollinger Bands (20, 2) from daily data
    close_daily = df_daily['close'].values
    sma20 = pd.Series(close_daily).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close_daily).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    
    # Align indicators to daily timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    upper_bb_aligned = align_htf_to_ltf(prices, df_daily, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_daily, lower_bb)
    sma20_aligned = align_htf_to_ltf(prices, df_daily, sma20)
    
    # Calculate 20-day volume average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or np.isnan(sma20_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: only trade in direction of weekly trend
        bullish_trend = close[i] > ema50_aligned[i]
        bearish_trend = close[i] < ema50_aligned[i]
        
        if position == 0:
            # Long: Price breaks above upper Bollinger Band with volume, in bullish trend
            if (close[i] > upper_bb_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20[i] and 
                bullish_trend):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Bollinger Band with volume, in bearish trend
            elif (close[i] < lower_bb_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20[i] and 
                  bearish_trend):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to middle Bollinger Band (mean reversion)
            if position == 1:
                if close[i] < sma20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > sma20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1D_Bollinger20_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0