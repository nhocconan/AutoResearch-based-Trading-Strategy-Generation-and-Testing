#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 6h Bollinger Bands (20, 2)
    close_series = pd.Series(close)
    sma_20 = close_series.rolling(window=20, min_periods=20).mean()
    std_20 = close_series.rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + (std_20 * 2)
    lower_bb = sma_20 - (std_20 * 2)
    
    # Calculate weekly ATR for volatility filter
    high_1w = pd.Series(df_1w['high'].values)
    low_1w = pd.Series(df_1w['low'].values)
    close_1w = pd.Series(df_1w['close'].values)
    tr1 = high_1w - low_1w
    tr2 = abs(high_1w - close_1w.shift(1))
    tr3 = abs(low_1w - close_1w.shift(1))
    tr_1w = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = tr_1w.rolling(window=14, min_periods=14).mean()
    
    # Calculate weekly EMA for trend filter
    ema_50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean()
    
    # Align all data to 6h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, prices, upper_bb.values)
    lower_bb_aligned = align_htf_to_ltf(prices, prices, lower_bb.values)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w.values)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w.values)
    
    # Calculate 6-day ATR for volatility threshold (approximate)
    high_6d = pd.Series(high)
    low_6d = pd.Series(low)
    close_6d = pd.Series(close)
    tr1_6d = high_6d - low_6d
    tr2_6d = abs(high_6d - close_6d.shift(1))
    tr3_6d = abs(low_6d - close_6d.shift(1))
    tr_6d = pd.concat([tr1_6d, tr2_6d, tr3_6d], axis=1).max(axis=1)
    atr_6d = tr_6d.rolling(window=6, min_periods=6).mean()
    atr_6d_aligned = align_htf_to_ltf(prices, prices, atr_6d.values)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or
            np.isnan(atr_1w_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(atr_6d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when 6d ATR > 1.5x weekly ATR
        volatility_filter = atr_6d_aligned[i] > (atr_1w_aligned[i] * 1.5)
        
        # Trend filter: only long when price > weekly EMA50, short when price < weekly EMA50
        long_trend = close[i] > ema_50_1w_aligned[i]
        short_trend = close[i] < ema_50_1w_aligned[i]
        
        # Entry conditions: Bollinger Band breakouts with volatility and trend confirmation
        if position == 0:
            # Long when price breaks above upper BB with volatility and uptrend
            if close[i] > upper_bb_aligned[i] and volatility_filter and long_trend:
                position = 1
                signals[i] = position_size
            # Short when price breaks below lower BB with volatility and downtrend
            elif close[i] < lower_bb_aligned[i] and volatility_filter and short_trend:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price returns to middle (SMA20) or shows reversal
            # Approximate SMA20 from recent closes
            sma_20_current = close[max(0, i-19):i+1].mean()
            if close[i] < sma_20_current:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price returns to middle (SMA20) or shows reversal
            sma_20_current = close[max(0, i-19):i+1].mean()
            if close[i] > sma_20_current:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_Bollinger_Breakout_Volatility_Filter"
timeframe = "6h"
leverage = 1.0