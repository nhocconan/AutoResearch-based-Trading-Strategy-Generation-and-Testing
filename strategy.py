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
    
    # Get 4h HTF data once before loop (primary timeframe is 4h per instructions)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period) - proven structure
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    upper_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian to 4h (same timeframe, but using completed bar alignment)
    upper_20_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_20_4h)
    lower_20_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_20_4h)
    
    # Get 1d HTF data for weekly pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior week (using 1d data)
    # Weekly high/low/close from 5 trading days ago (prior week)
    weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(5).values
    weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(5).values
    weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().shift(5).values
    
    # Weekly pivot: (H+L+C)/3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Weekly R1: 2*P - L
    weekly_r1 = 2 * weekly_pivot - weekly_low
    # Weekly S1: 2*P - H
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly pivot levels to 4h
    weekly_pivot_4h = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_4h = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_4h = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    # Get 1w HTF data for major trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1w EMA200 for major trend filter (proven BTC/ETH edge)
    ema_200_1w = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
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
    
    # Precompute hour filter for session (avoid low liquidity periods)
    hours = prices.index.hour
    # Focus on major sessions: 00-08 UTC (Asia), 12-20 UTC (Europe/US overlap)
    in_session = ((hours >= 0) & (hours <= 8)) | ((hours >= 12) & (hours <= 20))
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_4h_aligned[i]) or np.isnan(lower_20_4h_aligned[i]) or 
            np.isnan(weekly_pivot_4h[i]) or np.isnan(weekly_r1_4h[i]) or 
            np.isnan(weekly_s1_4h[i]) or np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(atr_14[i]) or np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 4h price breaks above 4h Donchian upper (20) - bullish breakout
        # 2. Price above weekly pivot (bullish bias from prior week)
        # 3. Price above 1w EMA200 (major uptrend filter)
        # 4. Volume confirmation: volume > 1.5x average (strict to reduce trades)
        # 5. Volatility filter: ATR > 0.3% of price (avoid extremely low volatility)
        if (close[i] > upper_20_4h_aligned[i] and
            close[i] > weekly_pivot_4h[i] and
            close[i] > ema_200_1w_aligned[i] and
            volume_ratio[i] > 1.5 and
            atr_14[i] > 0.003 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 4h price breaks below 4h Donchian lower (20) - bearish breakdown
        # 2. Price below weekly pivot (bearish bias from prior week)
        # 3. Price below 1w EMA200 (major downtrend filter)
        # 4. Volume confirmation: volume > 1.5x average
        # 5. Volatility filter: ATR > 0.3% of price
        elif (close[i] < lower_20_4h_aligned[i] and
              close[i] < weekly_pivot_4h[i] and
              close[i] < ema_200_1w_aligned[i] and
              volume_ratio[i] > 1.5 and
              atr_14[i] > 0.003 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_1d_WeeklyPivot_1w_EMA200_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0