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
    
    # Get daily data for ATR and price range calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily ATR (14-period) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period
    tr2[0] = np.nan  # No previous close
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR calculation with proper smoothing
    atr_1d = np.full(len(tr), np.nan)
    for i in range(14, len(tr)):
        if i == 14:
            atr_1d[i] = np.mean(tr[1:i+1])  # Initial SMA
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14  # Wilder's smoothing
    
    # Calculate daily price range (high - low)
    daily_range = high_1d - low_1d
    
    # Calculate range ratio: current daily range / ATR
    range_ratio = np.full(len(daily_range), np.nan)
    for i in range(len(daily_range)):
        if not np.isnan(atr_1d[i]) and atr_1d[i] > 0:
            range_ratio[i] = daily_range[i] / atr_1d[i]
    
    # Align ATR and range ratio to 12h timeframe
    atr_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    range_ratio_12h = align_htf_to_ltf(prices, df_1d, range_ratio)
    
    # Calculate 12-period EMA of close for trend filter
    close_series = pd.Series(close)
    ema_12 = close_series.ewm(span=12, adjust=False, min_periods=12).values
    
    # Calculate 12-period standard deviation for volatility bands
    rolling_std = pd.Series(close).rolling(window=12, min_periods=12).std().values
    
    # Dynamic volatility bands: EMA ± (1.5 * std dev)
    upper_band = ema_12 + 1.5 * rolling_std
    lower_band = ema_12 - 1.5 * rolling_std
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(range_ratio_12h[i]) or np.isnan(ema_12[i]) or 
            np.isnan(upper_band[i]) or np.isnan(lower_band[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: trade only when range is expanded (trending market)
        vol_expansion = range_ratio_12h[i] > 1.2
        
        # Trend filter: price above/below EMA
        price_above_ema = close[i] > ema_12[i]
        price_below_ema = close[i] < ema_12[i]
        
        # Entry conditions: volatility expansion + price outside volatility bands
        long_entry = vol_expansion and price_above_ema and close[i] > upper_band[i]
        short_entry = vol_expansion and price_below_ema and close[i] < lower_band[i]
        
        # Exit conditions: price returns to EMA or volatility contracts
        long_exit = close[i] < ema_12[i] or range_ratio_12h[i] < 0.8
        short_exit = close[i] > ema_12[i] or range_ratio_12h[i] < 0.8
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_volatility_breakout_v1"
timeframe = "12h"
leverage = 1.0