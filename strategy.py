# 12h ATR Breakout with 1d Trend Filter and Volume Confirmation
# Trend-following strategy that buys breakouts above ATR-based channels in the direction of the daily trend.
# Uses ATR(14) to set dynamic entry/exit levels, reducing whipsaws in choppy markets.
# Daily EMA34 filters trend direction to avoid counter-trend trades.
# Volume confirmation ensures breakouts have institutional participation.
# Designed for 12h timeframe to capture multi-day moves while avoiding overtrading.
# Target: 12-37 trades/year (50-150 total over 4 years) with low turnover to minimize fee drag.

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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_period = 34
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_period + 1)) + 
                         ema_1d[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Calculate ATR(14) for dynamic channels
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First TR is just high-low
    atr = np.full(n, np.nan)
    if n >= atr_period:
        atr[atr_period - 1] = np.mean(tr[:atr_period])
        for i in range(atr_period, n):
            atr[i] = (tr[i] * (1 / atr_period) + atr[i - 1] * (1 - 1 / atr_period))
    
    # Calculate upper and lower channels (using ATR multiplier)
    atr_mult = 2.0
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    for i in range(atr_period, n):
        upper_channel[i] = close[i - 1] + atr[i] * atr_mult
        lower_channel[i] = close[i - 1] - atr[i] * atr_mult
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    # Align 1d EMA to 12h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need ATR, channels, EMA34, and volume MA20
    start_idx = max(atr_period, ema_period - 1, 19) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: price breaks above upper channel with 1d EMA34 uptrend and volume filter
            if (price > upper_channel[i] and 
                price > ema_1d_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: price breaks below lower channel with 1d EMA34 downtrend and volume filter
            elif (price < lower_channel[i] and 
                  price < ema_1d_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below lower channel
            if price < lower_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price closes above upper channel
            if price > upper_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_ATRBreakout_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0