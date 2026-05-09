# !/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyTrend_WeeklyVolatilityBreakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and volatility filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA21 for trend filter
    ema21_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Weekly ATR(14) for volatility filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = np.maximum(high_1w[1:] - low_1w[1:], np.abs(high_1w[1:] - close_1w[:-1]), np.abs(low_1w[1:] - close_1w[:-1]))
    tr = np.concatenate([[high_1w[0] - low_1w[0]], tr1])
    atr14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily Donchian(20) breakout levels
    donch_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donch_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align weekly indicators to daily
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    atr14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr14_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Need enough data for Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(ema21_1w_aligned[i]) or np.isnan(atr14_1w_aligned[i]) or
            np.isnan(donch_high_20[i]) or np.isnan(donch_low_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend = ema21_1w_aligned[i]
        atr = atr14_1w_aligned[i]
        upper = donch_high_20[i]
        lower = donch_low_20[i]
        
        if position == 0:
            # Enter long: break above upper band with volatility expansion and above weekly trend
            if close[i] > upper and atr > 0 and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Enter short: break below lower band with volatility expansion and below weekly trend
            elif close[i] < lower and atr > 0 and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below lower band (mean reversion)
            if close[i] < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above upper band (mean reversion)
            if close[i] > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals