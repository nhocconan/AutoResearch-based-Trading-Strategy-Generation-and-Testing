#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with weekly trend filter and daily mean reversion
# - Weekly EMA(20) defines trend direction (long when price > weekly EMA20, short when price < weekly EMA20)
# - Daily RSI(14) for mean reversion entries: long when RSI < 30 in weekly uptrend, short when RSI > 70 in weekly downtrend
# - Exit on opposite RSI extreme (RSI > 70 for long, RSI < 30 for short) or weekly trend reversal
# - Volume confirmation: current 6h volume > 1.5x 20-period average of 6h volume
# - Position size: 0.25 (25%) to manage drawdown
# - Designed to work in both bull and bear markets by following higher timeframe trend while exploiting mean reversion
# - Target: 15-30 trades/year to avoid excessive fee drag

name = "6h_WeeklyTrend_DailyRSI_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly EMA(20) for trend direction
    ema_20_weekly = pd.Series(df_weekly['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_20_weekly)
    
    # Get daily data for RSI
    df_daily = get_htf_data(prices, '1d')
    
    # Daily RSI(14)
    delta = pd.Series(df_daily['close'].values).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_daily = 100 - (100 / (1 + rs))
    rsi_daily_values = rsi_daily.values
    rsi_daily_aligned = align_htf_to_ltf(prices, df_daily, rsi_daily_values)
    
    # 6h volume average (20-period)
    vol_6h = volume
    vol_ma_6h = pd.Series(vol_6h).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute time of day filter (optional: avoid low volatility periods)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Optional: time filter (uncomment if needed)
        # if hours[i] < 0 or hours[i] > 23:  # trade all hours
        #     signals[i] = 0.0
        #     continue
            
        # Skip if any required data is NaN
        if (np.isnan(ema_20_weekly_aligned[i]) or 
            np.isnan(rsi_daily_aligned[i]) or 
            np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 6h volume > 1.5x average
        volume_filter = vol_ma_6h[i] > 0 and volume[i] > 1.5 * vol_ma_6h[i]
        
        if position == 0:
            # Look for long entry: weekly uptrend (price > weekly EMA20) + oversold daily RSI + volume
            if close[i] > ema_20_weekly_aligned[i] and rsi_daily_aligned[i] < 30 and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: weekly downtrend (price < weekly EMA20) + overbought daily RSI + volume
            elif close[i] < ema_20_weekly_aligned[i] and rsi_daily_aligned[i] > 70 and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on overbought RSI or weekly trend reversal
            if rsi_daily_aligned[i] > 70 or close[i] < ema_20_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on oversold RSI or weekly trend reversal
            if rsi_daily_aligned[i] < 30 or close[i] > ema_20_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals