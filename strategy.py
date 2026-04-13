#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R with 1d trend filter and volume spike
    # Long: Williams %R < -80 (oversold) + 1d close > 1d EMA50 + volume > 2.0x 20-period 6h average
    # Short: Williams %R > -20 (overbought) + 1d close < 1d EMA50 + volume > 2.0x 20-period 6h average
    # Uses discrete sizing (0.25) to minimize fee drag and ATR-based stoploss
    # Target: 12-37 trades/year to stay within 6h optimal range (50-150 total over 4 years)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R on 6h data (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Calculate volume average for confirmation (using 6h data)
    vol_avg_20_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    # Calculate ATR using true range approximation for 6h timeframe
    atr_6h = np.zeros(n)
    for i in range(1, n):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        if i < 14:
            atr_6h[i] = tr  # Simple average for warmup
        else:
            atr_6h[i] = 0.93 * atr_6h[i-1] + 0.07 * tr  # Wilder's smoothing
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_avg_20_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 2.0x 20-period average
        volume_confirmed = volume[i] > 2.0 * vol_avg_20_6h[i]
        
        # Trend filter: 6h close above/below EMA50 (from 1d)
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Williams %R conditions: oversold/overbought with volume and trend
        long_signal = (williams_r[i] < -80) and volume_confirmed and uptrend
        short_signal = (williams_r[i] > -20) and volume_confirmed and downtrend
        
        # Stoploss: 2x ATR below/above entry
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - 2.0 * atr_6h[i]
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + 2.0 * atr_6h[i]
        
        # Execute signals
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            elif position == -1:
                signals[i] = -position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            else:
                signals[i] = 0.0
                entry_price[i] = np.nan
    
    return signals

name = "6h_1d_williams_r_volume_trend_v1"
timeframe = "6h"
leverage = 1.0