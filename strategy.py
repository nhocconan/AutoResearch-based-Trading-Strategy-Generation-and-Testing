#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for weekly trend and ATR
    df_1d = get_htf_data(prices, '1d')
    
    # Weekly trend: 50-period EMA on daily data
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Daily ATR for volatility filter (14-period)
    atr14_1d = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))).rolling(window=14, min_periods=14).mean().values
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Volume: daily volume ratio to 20-day average
    vol_series = pd.Series(volume)
    avg_vol20 = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    vol_ratio = vol_series / np.where(avg_vol20 == 0, 1e-10, avg_vol20)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio.values)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after sufficient data
    start = max(50, 20)  # 50 for EMA, 20 for volume average
    
    for i in range(start, n):
        # Skip if critical data is NaN
        if (np.isnan(ema50_aligned[i]) or np.isnan(atr14_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr14_aligned[i]
        vol_ratio = vol_ratio_aligned[i]
        
        if position == 0:
            # Long: price above weekly EMA50, low volatility, high volume
            if price > ema50_aligned[i] and atr < 0.02 * price and vol_ratio > 1.5:
                position = 1
                signals[i] = position_size
            # Short: price below weekly EMA50, low volatility, high volume
            elif price < ema50_aligned[i] and atr < 0.02 * price and vol_ratio > 1.5:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below weekly EMA50 or volatility spikes
            if price < ema50_aligned[i] or atr > 0.04 * price:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above weekly EMA50 or volatility spikes
            if price > ema50_aligned[i] or atr > 0.04 * price:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_EMA50_Volume_Volatility_Filter"
timeframe = "1d"
leverage = 1.0