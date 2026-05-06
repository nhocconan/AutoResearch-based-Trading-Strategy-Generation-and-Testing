#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day Choppiness Index regime filter with 4-hour EMA trend filter
# Long when CHOP(14) > 61.8 (ranging) and price > 4h EMA50 with volume confirmation
# Short when CHOP(14) > 61.8 (ranging) and price < 4h EMA50 with volume confirmation
# Uses daily Choppiness Index to identify ranging markets, 4h EMA for direction
# Designed to work in ranging markets via mean reversion and avoid trending markets
# Target: 20-30 trades per year (80-120 over 4 years) with 0.25 position sizing

name = "12h_1dChoppiness_4hEMA50_Volume"
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
    
    # Calculate 1-day Choppiness Index (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range calculation
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = df_1d['high'].rolling(window=14, min_periods=14).max().values
    ll = df_1d['low'].rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index formula
    chop = 100 * np.log10(sum_tr / (atr * 14)) / np.log10(14)
    
    # Align Choppiness Index to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 4-hour EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    ema_50 = pd.Series(df_4h['close']).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Volume confirmation: >1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(chop_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Chop > 61.8 indicates ranging market (mean reversion opportunity)
        if chop_aligned[i] > 61.8:
            if position == 0:
                # Long when price above EMA50 in ranging market
                if close[i] > ema_50_aligned[i] and volume_filter[i]:
                    signals[i] = 0.25
                    position = 1
                # Short when price below EMA50 in ranging market
                elif close[i] < ema_50_aligned[i] and volume_filter[i]:
                    signals[i] = -0.25
                    position = -1
            elif position == 1:
                # Exit long when price crosses below EMA50
                if close[i] < ema_50_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short when price crosses above EMA50
                if close[i] > ema_50_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # In trending market (Chop <= 61.8), stay flat
            if position != 0:
                signals[i] = 0.0
                position = 0
    
    return signals