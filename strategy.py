#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_ema_bounce_v1"
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
    
    # Get weekly data for EMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Weekly EMA(21) - trend filter
    ema_21 = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21)
    
    # Get daily data for pullback measurement
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Daily RSI(14) for oversold/overbought
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # Daily ATR(14) for volatility
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    atr_values = atr.values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_values)
    
    # Weekly high/low for dynamic support/resistance
    weekly_high = df_1w['high'].rolling(window=4, min_periods=4).max().values
    weekly_low = df_1w['low'].rolling(window=4, min_periods=4).min().values
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(ema_21_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(weekly_high_aligned[i]) or 
            np.isnan(weekly_low_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: price near weekly low, RSI oversold, above weekly EMA
        near_support = close[i] <= weekly_low_aligned[i] + (atr_1d_aligned[i] * 0.5)
        rsi_oversold = rsi_1d_aligned[i] < 35
        above_ema = close[i] > ema_21_aligned[i]
        long_signal = near_support and rsi_oversold and above_ema
        
        # Short: price near weekly high, RSI overbought, below weekly EMA
        near_resistance = close[i] >= weekly_high_aligned[i] - (atr_1d_aligned[i] * 0.5)
        rsi_overbought = rsi_1d_aligned[i] > 65
        below_ema = close[i] < ema_21_aligned[i]
        short_signal = near_resistance and rsi_overbought and below_ema
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and close[i] < ema_21_aligned[i]:
            # Exit long if price falls below weekly EMA
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > ema_21_aligned[i]:
            # Exit short if price rises above weekly EMA
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals