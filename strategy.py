#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly KAMA trend with daily RSI mean reversion and volume confirmation.
# Long when weekly KAMA turns up AND daily RSI < 30 (oversold) AND daily volume > 1.5x 20-day average.
# Short when weekly KAMA turns down AND daily RSI > 70 (overbought) AND daily volume > 1.5x 20-day average.
# Exit when RSI crosses 50 (mean reversion complete).
# Weekly KAMA provides adaptive trend filter, daily RSI captures mean reversion in both bull/bear markets,
# volume confirms participation. Target: 15-25 trades/year per symbol (60-100 total) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for KAMA
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly KAMA (adaptive moving average)
    def kama(close, period=10, fast=2, slow=30):
        n = len(close)
        if n < period:
            return np.full(n, np.nan)
        
        # Efficiency Ratio
        change = np.abs(np.diff(close, period))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.zeros(n)
        er[period:] = change / volatility
        er[volatility == 0] = 0
        
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # KAMA calculation
        kama_vals = np.full(n, np.nan)
        kama_vals[period-1] = close[period-1]
        for i in range(period, n):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals
    
    kama_weekly = kama(close_weekly, 10, 2, 30)
    kama_up = np.zeros_like(kama_weekly, dtype=bool)
    kama_up[1:] = kama_weekly[1:] > kama_weekly[:-1]
    kama_down = np.zeros_like(kama_weekly, dtype=bool)
    kama_down[1:] = kama_weekly[1:] < kama_weekly[:-1]
    
    # Load daily data ONCE for RSI and volume
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    close_daily = df_daily['close'].values
    volume_daily = df_daily['volume'].values
    
    # Calculate daily RSI (14-period)
    def rsi(close, period=14):
        n = len(close)
        if n < period + 1:
            return np.full(n, np.nan)
        
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        
        # First average
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        # Wilder's smoothing
        for i in range(period + 1, n):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    rsi_daily = rsi(close_daily, 14)
    
    # Calculate 20-day average volume
    vol_ma_20 = np.full_like(volume_daily, np.nan)
    for i in range(19, len(volume_daily)):
        vol_ma_20[i] = np.mean(volume_daily[i-19:i+1])
    
    # Align indicators to daily timeframe
    kama_up_aligned = align_htf_to_ltf(prices, df_weekly, kama_up)
    kama_down_aligned = align_htf_to_ltf(prices, df_weekly, kama_down)
    rsi_aligned = align_htf_to_ltf(prices, df_daily, rsi_daily)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current daily volume vs 20-day average
        daily_volume_aligned = align_htf_to_ltf(prices, df_daily, volume_daily)
        volume_ratio = daily_volume_aligned[i] / vol_ma_20_aligned[i] if vol_ma_20_aligned[i] > 0 else 0
        
        if position == 0:
            # Look for mean reversion entries with volume confirmation and trend filter
            # Long: RSI oversold AND weekly KAMA turning up AND volume > 1.5x average
            if (rsi_aligned[i] < 30 and 
                kama_up_aligned[i] and 
                volume_ratio > 1.5):
                position = 1
                signals[i] = position_size
            # Short: RSI overbought AND weekly KAMA turning down AND volume > 1.5x average
            elif (rsi_aligned[i] > 70 and 
                  kama_down_aligned[i] and 
                  volume_ratio > 1.5):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI crosses 50 (mean reversion complete)
            if rsi_aligned[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI crosses 50 (mean reversion complete)
            if rsi_aligned[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_KAMA_RSI_MeanReversion_Volume_v1"
timeframe = "1d"
leverage = 1.0