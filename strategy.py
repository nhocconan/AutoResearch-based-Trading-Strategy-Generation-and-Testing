#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily pivot points from prior day (standard formula)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Prior day's OHLC for pivot calculation
    prev_high = np.concatenate([[daily_high[0]], daily_high[:-1]])
    prev_low = np.concatenate([[daily_low[0]], daily_low[:-1]])
    prev_close = np.concatenate([[daily_close[0]], daily_close[:-1]])
    
    # Standard pivot: (H+L+C)/3
    daily_pivot = (prev_high + prev_low + prev_close) / 3.0
    # R1: 2*P - L
    daily_r1 = 2 * daily_pivot - prev_low
    # S1: 2*P - H
    daily_s1 = 2 * daily_pivot - prev_high
    # R2: P + (H-L)
    daily_r2 = daily_pivot + (prev_high - prev_low)
    # S2: P - (H-L)
    daily_s2 = daily_pivot - (prev_high - prev_low)
    
    # Align daily pivot levels to 4h
    daily_pivot_4h = align_htf_to_ltf(prices, df_1d, daily_pivot)
    daily_r1_4h = align_htf_to_ltf(prices, df_1d, daily_r1)
    daily_s1_4h = align_htf_to_ltf(prices, df_1d, daily_s1)
    daily_r2_4h = align_htf_to_ltf(prices, df_1d, daily_r2)
    daily_s2_4h = align_htf_to_ltf(prices, df_1d, daily_s2)
    
    # Get 1w HTF data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA200 for long-term trend
    weekly_close = df_1w['close'].values
    ema_200_1w = pd.Series(weekly_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate 4h ATR(14) for volatility filter and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    # Precompute session filter (00-24 UTC for 4h - less restrictive)
    hours = prices.index.hour
    in_session = (hours >= 0) & (hours <= 23)  # Always true for 4h, kept for structure
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(daily_pivot_4h[i]) or np.isnan(daily_r1_4h[i]) or np.isnan(daily_s1_4h[i]) or
            np.isnan(daily_r2_4h[i]) or np.isnan(daily_s2_4h[i]) or np.isnan(ema_200_1w_aligned[i]) or
            np.isnan(atr_14[i]) or np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Price breaks above daily R1 (bullish breakout)
        # 2. Price above weekly EMA200 (bullish long-term trend)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        if (close[i] > daily_r1_4h[i] and
            close[i] > ema_200_1w_aligned[i] and
            volume_ratio[i] > 1.5 and
            atr_14[i] > 0.005 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below daily S1 (bearish breakdown)
        # 2. Price below weekly EMA200 (bearish long-term trend)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility filter: ATR > 0.5% of price
        elif (close[i] < daily_s1_4h[i] and
              close[i] < ema_200_1w_aligned[i] and
              volume_ratio[i] > 1.5 and
              atr_14[i] > 0.005 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_DailyPivot_R1S1_1wEMA200_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0