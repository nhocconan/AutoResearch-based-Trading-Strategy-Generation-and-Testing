#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d daily Pivot Point R1/S1 breakout with volume confirmation and ATR filter
# Uses weekly trend filter to align with higher timeframe momentum
# Target: 20-40 trades per year, works in both bull and bear markets via trend alignment
# Focus on BTC/ETH as primary targets with volume confirmation to avoid false breakouts

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA34 for trend direction
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Load daily data for pivot points and ATR (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Pivot Point, R1, S1
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align daily levels to 1d timeframe (no additional delay needed for pivot points)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 1d ATR for volatility filter and stop consideration
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1d price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: 20-period average (approx 1 month of daily bars)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        weekly_trend_up = ema_34_1w_aligned[i] > close_1d[-1] if len(close_1d) > 0 else False  # Simplified trend check
        
        if position == 0:
            # Long: break above R1 with volume, sufficient volatility, and weekly uptrend
            if price > r1_1d_aligned[i] and vol > 1.8 * vol_ma[i] and atr_1d_aligned[i] > 0 and weekly_trend_up:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume, sufficient volatility, and weekly downtrend
            elif price < s1_1d_aligned[i] and vol > 1.8 * vol_ma[i] and atr_1d_aligned[i] > 0 and not weekly_trend_up:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below S1 or volatility drops significantly
            if price < s1_1d_aligned[i] or vol < 0.7 * vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above R1 or volatility drops significantly
            if price > r1_1d_aligned[i] or vol < 0.7 * vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_PivotPoint_R1S1_Breakout_Volume_WeeklyTrendFilter"
timeframe = "1d"
leverage = 1.0