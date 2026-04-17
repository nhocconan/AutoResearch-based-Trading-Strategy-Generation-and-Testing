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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly EMA50 for trend direction
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate daily ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    tr = np.concatenate([[np.nan], tr2])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_1d = align_htf_to_ltf(prices, df_1d, atr14)
    
    # Calculate daily Bollinger Bands width for regime filter
    sma20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    bb_width = (upper_bb - lower_bb) / sma20
    bb_width_1d = align_htf_to_ltf(prices, df_1d, bb_width)
    
    # Daily Donchian breakout levels (20-period)
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_high_1d = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_1d = align_htf_to_ltf(prices, df_1d, donch_low)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need sufficient data for weekly EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_1d[i]) or np.isnan(atr14_1d[i]) or 
            np.isnan(bb_width_1d[i]) or np.isnan(donch_high_1d[i]) or 
            np.isnan(donch_low_1d[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: avoid extremely low volatility (choppy) markets
        vol_filter = bb_width_1d[i] > 0.02  # Only trade when Bollinger width > 2%
        
        # Trend filter: price relative to weekly EMA50
        price_above_ema = close[i] > ema50_1d[i]
        price_below_ema = close[i] < ema50_1d[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high with trend and volatility
            if (close[i] > donch_high_1d[i] and price_above_ema and vol_filter):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low with trend and volatility
            elif (close[i] < donch_low_1d[i] and price_below_ema and vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below Donchian low or trend reverses
            if close[i] < donch_low_1d[i] or close[i] < ema50_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above Donchian high or trend reverses
            if close[i] > donch_high_1d[i] or close[i] > ema50_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyEMA50_Donchian20_VolFilter"
timeframe = "1d"
leverage = 1.0