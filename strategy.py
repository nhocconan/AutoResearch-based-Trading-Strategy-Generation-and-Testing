#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Keltner Channel breakout with volume confirmation and monthly trend filter.
# Long when price breaks above upper KC (EMA20 + 2*ATR) with volume > 1.5x average and monthly trend up.
# Short when price breaks below lower KC (EMA20 - 2*ATR) with volume > 1.5x average and monthly trend down.
# Exit when price crosses back below/above EMA20.
# Uses weekly KC for breakout signals, volume for confirmation, monthly EMA50 for trend filter.
# Target: 10-25 trades/year to avoid fee drag. Works in bull/bear via trend-filtered breakouts.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Keltner Channel
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Get monthly data for trend filter (EMA50)
    df_monthly = get_htf_data(prices, '1M')
    if len(df_monthly) < 50:
        return np.zeros(n)
    
    close_monthly = df_monthly['close'].values
    
    # Calculate ATR(14) for weekly data
    def calculate_atr(high, low, close, period):
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        atr = np.full_like(tr, np.nan, dtype=float)
        for i in range(period, len(tr)):
            if i == period:
                atr[i] = np.nanmean(tr[1:period+1])
            else:
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_weekly = calculate_atr(high_weekly, low_weekly, close_weekly, 14)
    
    # Calculate EMA(20) for weekly data
    def calculate_ema(data, period):
        ema = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return ema
        multiplier = 2 / (period + 1)
        ema[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            ema[i] = (data[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        return ema
    
    ema20_weekly = calculate_ema(close_weekly, 20)
    
    # Calculate Keltner Channel: EMA20 ± 2*ATR
    kc_upper = ema20_weekly + 2 * atr_weekly
    kc_lower = ema20_weekly - 2 * atr_weekly
    
    # Calculate EMA(50) for monthly trend filter
    ema50_monthly = calculate_ema(close_monthly, 50)
    
    # Get volume MA(20) for daily data
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Align indicators to daily timeframe
    kc_upper_aligned = align_htf_to_ltf(prices, df_weekly, kc_upper)
    kc_lower_aligned = align_htf_to_ltf(prices, df_weekly, kc_lower)
    ema50_aligned = align_htf_to_ltf(prices, df_monthly, ema50_monthly)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 30-period weekly data and 20-period volume MA
    start_idx = max(30, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kc_upper_aligned[i]) or np.isnan(kc_lower_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filter: monthly EMA50 slope
        trend_up = ema50_aligned[i] > ema50_aligned[i-1] if i > 0 else False
        trend_down = ema50_aligned[i] < ema50_aligned[i-1] if i > 0 else False
        
        if position == 0:
            # Long: price breaks above KC upper with volume and trend up
            if price > kc_upper_aligned[i] and vol_filter and trend_up:
                signals[i] = size
                position = 1
            # Short: price breaks below KC lower with volume and trend down
            elif price < kc_lower_aligned[i] and vol_filter and trend_down:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below EMA20 (using KC middle as proxy)
            if price < kc_upper_aligned[i] - atr_weekly[i]:  # EMA20 ≈ KC middle
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above EMA20
            if price > kc_lower_aligned[i] + atr_weekly[i]:  # EMA20 ≈ KC middle
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KeltnerChannel_Breakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0