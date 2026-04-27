#!/usr/bin/env python3
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
    
    # Get 1d data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR (14-period) for volatility filter
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_14 = np.full(len(tr), np.nan)
    for i in range(13, len(tr)):
        atr_14[i] = np.mean(tr[i-13:i+1])
    
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate daily EMA (50-period) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate daily range for volatility filter
    daily_range = high_1d - low_1d
    range_ma_10 = np.full(len(daily_range), np.nan)
    for i in range(9, len(daily_range)):
        range_ma_10[i] = np.mean(daily_range[i-9:i+1])
    range_ma_aligned = align_htf_to_ltf(prices, df_1d, range_ma_10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need ATR (14), EMA (50), range MA (10)
    start_idx = max(14, 50, 10)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_14_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(range_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_ma_5 = np.mean(volume[i-5:i+1]) if i >= 5 else np.mean(volume[:i+1])
        
        # Volatility filter: trade only when volatility is elevated
        vol_filter = atr_14_aligned[i] > 0.5 * range_ma_aligned[i]
        
        # Trend filter from daily EMA
        bullish_trend = price > ema_50_aligned[i]
        bearish_trend = price < ema_50_aligned[i]
        
        # Volume confirmation: above average volume
        vol_confirm = vol_now > vol_ma_5
        
        if position == 0:
            # Long: price above EMA50 + volatility + volume
            if bullish_trend and vol_filter and vol_confirm:
                signals[i] = size
                position = 1
            # Short: price below EMA50 + volatility + volume
            elif bearish_trend and vol_filter and vol_confirm:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below EMA50 or volatility drops
            if not bullish_trend or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above EMA50 or volatility drops
            if not bearish_trend or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_EMA50_ATR_Volume_Filter"
timeframe = "1d"
leverage = 1.0