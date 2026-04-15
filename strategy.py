#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot (R1/S1) breakout with volume confirmation and ATR filter
# Uses weekly trend as regime filter: only trade long when price > weekly EMA20, short when price < weekly EMA20
# This reduces whipsaw in sideways markets and focuses on stronger moves
# Target: 20-50 trades/year (80-200 over 4 years) to avoid fee drag
# Works in bull (breakouts with trend) and bear (mean reversion at extremes with trend filter)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly HTF data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_volume = df_1d['volume'].values
    
    weekly_close = df_1w['close'].values
    
    # Calculate daily pivot points (standard floor trader's pivots)
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    # R2 = P + (H - L)
    # S2 = P - (H - L)
    pivot = (daily_high + daily_low + daily_close) / 3.0
    r1 = 2 * pivot - daily_low
    s1 = 2 * pivot - daily_high
    
    # Calculate weekly EMA20 for trend filter
    weekly_ema20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = pd.Series(daily_high - daily_low)
    tr2 = pd.Series(np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]])))
    tr3 = pd.Series(np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 1d timeframe (no additional delay needed for pivots/EMA)
    pivot_1d = align_htf_to_ltf(prices, df_1d, pivot)
    r1_1d = align_htf_to_ltf(prices, df_1d, r1)
    s1_1d = align_htf_to_ltf(prices, df_1d, s1)
    atr_14_1d = align_htf_to_ltf(prices, df_1d, atr_14)
    weekly_ema20_1d = align_htf_to_ltf(prices, df_1w, weekly_ema20)
    
    # Calculate 1d volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_1d[i]) or np.isnan(r1_1d[i]) or np.isnan(s1_1d[i]) or 
            np.isnan(atr_14_1d[i]) or np.isnan(weekly_ema20_1d[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in direction of weekly trend
        weekly_trend_up = close[i] > weekly_ema20_1d[i]
        weekly_trend_down = close[i] < weekly_ema20_1d[i]
        
        # Entry conditions:
        # Long: price breaks above R1 with volume confirmation AND weekly uptrend
        # Short: price breaks below S1 with volume confirmation AND weekly downtrend
        # Volume confirmation: volume > 1.5x average
        # Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        # Discrete position sizing: 0.25
        
        # Long conditions: 1d breakout above R1 with weekly uptrend
        if (close[i] > r1_1d[i] and            # 1d price above R1 pivot
            weekly_trend_up and               # Weekly uptrend filter
            volume_ratio[i] > 1.5 and         # Volume confirmation
            atr_14_1d[i] > 0.005 * close[i]): # Volatility filter
            signals[i] = 0.25
            
        # Short conditions: 1d breakdown below S1 with weekly downtrend
        elif (close[i] < s1_1d[i] and          # 1d price below S1 pivot
              weekly_trend_down and            # Weekly downtrend filter
              volume_ratio[i] > 1.5 and        # Volume confirmation
              atr_14_1d[i] > 0.005 * close[i]): # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_Volume_WeeklyTrend_Filter"
timeframe = "1d"
leverage = 1.0