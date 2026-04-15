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
    
    # Get daily HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_volume = df_1d['volume'].values
    
    # Calculate 14-period daily ATR for volatility regime filter
    daily_close_prev = np.concatenate([[daily_close[0]], daily_close[:-1]])
    tr = np.maximum(daily_high - daily_low,
                    np.maximum(np.abs(daily_high - daily_close_prev),
                               np.abs(daily_low - daily_close_prev)))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    volatility_ratio = atr_14 / (atr_ma_50 + 1e-10)
    
    # Calculate 50-period daily EMA for trend filter
    ema_50 = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily RSI(14) for momentum filter
    delta = np.diff(daily_close, prepend=daily_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Align HTF indicators to 6h timeframe with proper delay
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50)
    rsi_14_6h = align_htf_to_ltf(prices, df_1d, rsi_14)
    volatility_ratio_6h = align_htf_to_ltf(prices, df_1d, volatility_ratio)
    
    # Calculate 6h Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    # Add minimum holding period to reduce trade frequency
    position = 0  # 0 = flat, 1 = long, -1 = short
    bars_since_entry = 0
    min_hold_bars = 12  # Minimum 12 bars (3 days for 6h) holding period
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_6h[i]) or np.isnan(rsi_14_6h[i]) or 
            np.isnan(volatility_ratio_6h[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        # Exit conditions: reverse signal or min hold period exceeded with opposing signal
        if position == 1 and bars_since_entry >= min_hold_bars:
            # Exit long if price breaks below Donchian low or RSI becomes oversold
            if close[i] < lowest_20[i] or rsi_14_6h[i] < 30:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1 and bars_since_entry >= min_hold_bars:
            # Exit short if price breaks above Donchian high or RSI becomes overbought
            if close[i] > highest_20[i] or rsi_14_6h[i] > 70:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25  # Maintain short
        else:
            # Entry logic
            if position == 0:  # Only enter when flat
                # Long conditions
                if (close[i] > ema_50_6h[i] and  # Uptrend filter
                    rsi_14_6h[i] < 70 and       # Not overbought
                    volatility_ratio_6h[i] > 0.8 and  # Avoid low volatility squeezes
                    close[i] > highest_20[i] and     # Donchian breakout
                    volume_ratio[i] > 1.5):        # Volume confirmation
                    signals[i] = 0.25
                    position = 1
                    bars_since_entry = 0
                    
                # Short conditions
                elif (close[i] < ema_50_6h[i] and   # Downtrend filter
                      rsi_14_6h[i] > 30 and       # Not oversold
                      volatility_ratio_6h[i] > 0.8 and  # Avoid low volatility squeezes
                      close[i] < lowest_20[i] and      # Donchian breakdown
                      volume_ratio[i] > 1.5):        # Volume confirmation
                    signals[i] = -0.25
                    position = -1
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
            else:
                # Maintain current position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_DailyEMA_RSI_Volume_Donchian_Breakout_v2"
timeframe = "6h"
leverage = 1.0