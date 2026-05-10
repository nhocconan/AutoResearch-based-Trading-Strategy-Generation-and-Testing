#!/usr/bin/env python3
# 1d_KAMA_Direction_With_RSI_Filter_And_Chop_Regime
# Hypothesis: On the daily timeframe, KAMA adapts to market conditions (trending vs ranging).
# In trending markets (Chop < 38.2), we follow KAMA direction with RSI filter for momentum confirmation.
# In ranging markets (Chop > 61.8), we mean-revert at Bollinger Bands with RSI extremes.
# This dual regime approach works in both bull and bear markets by adapting to market structure.
# Uses volume confirmation to avoid low-conviction signals.

name = "1d_KAMA_Direction_With_RSI_Filter_And_Chop_Regime"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend context
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate KAMA (adaptive moving average) on daily
    close_s = pd.Series(close)
    # Efficiency Ratio
    change = abs(close_s.diff(10))
    volatility = close_s.diff().abs().rolling(window=10).sum()
    er = change / volatility.replace(0, np.nan)
    # Smoothing constants
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Calculate Bollinger Bands
    bb_middle = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Calculate Choppiness Index (14-period)
    atr = np.zeros(n)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # True Range sum for denominator
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Choppiness Index
    chop = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(14)
    chop = np.nan_to_num(chop, nan=50.0)
    
    # Volume confirmation (20-day average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get weekly close for trend filter
    if len(df_1w) > 0:
        weekly_close = df_1w['close'].values
        weekly_close_aligned = align_htf_to_ltf(prices, df_1w, weekly_close)
    else:
        weekly_close_aligned = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (implicit), RSI (14), BB (20), Chop (14), Volume MA (20)
    start_idx = max(20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or np.isnan(chop[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Market regime based on Choppiness Index
        trending = chop[i] < 38.2  # Trending market
        ranging = chop[i] > 61.8   # Ranging market
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry conditions
            if trending and volume_confirm:
                # In trending market: follow KAMA direction with RSI momentum
                if close[i] > kama[i] and rsi[i] > 50 and rsi[i] < 70:
                    signals[i] = 0.25
                    position = 1
            elif ranging and volume_confirm:
                # In ranging market: mean revert at Bollinger Bands with RSI extremes
                if close[i] <= bb_lower[i] and rsi[i] < 30:
                    signals[i] = 0.25
                    position = 1
            
            # Short entry conditions
            if trending and volume_confirm:
                # In trending market: follow KAMA direction with RSI momentum
                if close[i] < kama[i] and rsi[i] < 50 and rsi[i] > 30:
                    signals[i] = -0.25
                    position = -1
            elif ranging and volume_confirm:
                # In ranging market: mean revert at Bollinger Bands with RSI extremes
                if close[i] >= bb_upper[i] and rsi[i] > 70:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit conditions
            exit_signal = False
            if trending:
                # Exit trend trade: KAMA cross or RSI overbought
                if close[i] < kama[i] or rsi[i] >= 70:
                    exit_signal = True
            else:  # ranging
                # Exit mean reversion: price reaches middle band or RSI neutral
                if close[i] >= bb_middle[i] or (rsi[i] >= 40 and rsi[i] <= 60):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions
            exit_signal = False
            if trending:
                # Exit trend trade: KAMA cross or RSI oversold
                if close[i] > kama[i] or rsi[i] <= 30:
                    exit_signal = True
            else:  # ranging
                # Exit mean reversion: price reaches middle band or RSI neutral
                if close[i] <= bb_middle[i] or (rsi[i] >= 40 and rsi[i] <= 60):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals