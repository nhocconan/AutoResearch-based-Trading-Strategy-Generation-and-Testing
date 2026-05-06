#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using weekly KAMA for trend direction and 1d RSI for mean reversion
# - Weekly KAMA trend filter: only take long when price > weekly KAMA, short when price < weekly KAMA
# - 1d RSI mean reversion: long when RSI < 30, short when RSI > 70
# - Volume confirmation: require volume > 1.5x 20-period average
# - Designed to work in both bull and bear markets by combining trend following with mean reversion
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "12h_WeeklyKAMA_1dRSI30_70_VolumeFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for KAMA trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on weekly close
    # ER (Efficiency Ratio) = |change| / sum(|changes|)
    # SC = [ER * (fastest - slowest) + slowest]^2
    # KAMA[i] = KAMA[i-1] + SC * (price[i] - KAMA[i-1])
    close_weekly = df_weekly['close'].values
    n_weekly = len(close_weekly)
    
    if n_weekly < 2:
        kama_weekly = np.full(n_weekly, np.nan)
    else:
        change = np.abs(np.diff(close_weekly))
        volatility = np.zeros(n_weekly)
        volatility[1:] = np.convolve(change, np.ones(10), mode='same')  # 10-period sum of absolute changes
        volatility[0] = change[0] if len(change) > 0 else 0
        
        # Avoid division by zero
        er = np.zeros(n_weekly)
        er[1:] = change[1:] / np.where(volatility[1:] == 0, 1, volatility[1:])
        er[0] = 0
        
        # Smoothing constants
        fastest = 2 / (2 + 1)   # 2-period EMA
        slowest = 2 / (30 + 1)  # 30-period EMA
        sc = (er * (fastest - slowest) + slowest) ** 2
        
        kama_weekly = np.zeros(n_weekly)
        kama_weekly[0] = close_weekly[0]
        for i in range(1, n_weekly):
            kama_weekly[i] = kama_weekly[i-1] + sc[i] * (close_weekly[i] - kama_weekly[i-1])
    
    # Align weekly KAMA to 12h timeframe
    kama_weekly_aligned = align_htf_to_ltf(prices, df_weekly, kama_weekly)
    
    # Get daily data for RSI
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 14:
        return np.zeros(n)
    
    # Calculate 14-period RSI on daily close
    close_daily = df_daily['close'].values
    delta = np.diff(close_daily)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(close_daily)
    avg_loss = np.zeros_like(close_daily)
    
    if len(gain) >= 14:
        avg_gain[13] = np.mean(gain[1:15])
        avg_loss[13] = np.mean(loss[1:15])
        
        for i in range(14, len(close_daily)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_daily = 100 - (100 / (1 + rs))
    
    # Align daily RSI to 12h timeframe
    rsi_daily_aligned = align_htf_to_ltf(prices, df_daily, rsi_daily)
    
    # Volume filters
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)  # Volume confirmation
    
    # Session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(kama_weekly_aligned[i]) or np.isnan(rsi_daily_aligned[i]) or 
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long signal: price above weekly KAMA (uptrend) AND RSI < 30 (oversold) AND volume confirmation
            if close[i] > kama_weekly_aligned[i] and rsi_daily_aligned[i] < 30 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short signal: price below weekly KAMA (downtrend) AND RSI > 70 (overbought) AND volume confirmation
            elif close[i] < kama_weekly_aligned[i] and rsi_daily_aligned[i] > 70 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below weekly KAMA (trend change) OR RSI > 70 (overbought)
            if close[i] < kama_weekly_aligned[i] or rsi_daily_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above weekly KAMA (trend change) OR RSI < 30 (oversold)
            if close[i] > kama_weekly_aligned[i] or rsi_daily_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals