#!/usr/bin/env python3
"""
6h_RSI_Divergence_Volume_HTFTrend
Hypothesis: On 6h timeframe, enter on bullish/bearish RSI divergence with volume confirmation and 12h EMA50 trend filter. RSI divergence catches reversals at extremes, volume confirms conviction, and HTF trend ensures alignment with higher timeframe momentum. Designed for 15-35 trades/year by requiring triple confluence. Works in bull/bear via EMA trend filter and divergence logic that adapts to market direction.
Primary timeframe: 6h, HTF: 12h for trend filter.
"""

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
    
    # Load 12h data ONCE before loop for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    close_12h_series = pd.Series(close_12h)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate RSI(14) on 6h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # Volume confirmation: volume > 1.8x 30-period median
    volume_series = pd.Series(volume)
    vol_median_30 = volume_series.rolling(window=30, min_periods=30).median().values
    volume_confirm = volume > (1.8 * vol_median_30)
    
    # Swing points for divergence detection (lookback 5 bars)
    def find_swing_highs(arr, lookback=5):
        highs = np.zeros_like(arr, dtype=bool)
        for i in range(lookback, len(arr) - lookback):
            window = arr[i-lookback:i+lookback+1]
            if arr[i] == np.max(window):
                highs[i] = True
        return highs
    
    def find_swing_lows(arr, lookback=5):
        lows = np.zeros_like(arr, dtype=bool)
        for i in range(lookback, len(arr) - lookback):
            window = arr[i-lookback:i+lookback+1]
            if arr[i] == np.min(window):
                lows[i] = True
        return lows
    
    # Find swing highs/lows for price and RSI
    price_swing_high = find_swing_highs(high, 5)
    price_swing_low = find_swing_lows(low, 5)
    rsi_swing_high = find_swing_highs(rsi, 5)
    rsi_swing_low = find_swing_lows(rsi, 5)
    
    # Detect divergences
    bullish_div = np.zeros(n, dtype=bool)  # price LL, RSI HL
    bearish_div = np.zeros(n, dtype=bool)  # price HH, RSI LH
    
    # Track recent swing points for divergence checking
    last_price_low = np.nan
    last_price_high = np.nan
    last_rsi_low = np.nan
    last_rsi_high = np.nan
    
    for i in range(10, n):
        if price_swing_low[i]:
            if not np.isnan(last_price_low) and low[i] < last_price_low and not np.isnan(last_rsi_low):
                if rsi[i] > last_rsi_low:
                    bullish_div[i] = True
            last_price_low = low[i]
            last_rsi_low = rsi[i]
        
        if price_swing_high[i]:
            if not np.isnan(last_price_high) and high[i] > last_price_high and not np.isnan(last_rsi_high):
                if rsi[i] < last_rsi_high:
                    bearish_div[i] = True
            last_price_high = high[i]
            last_rsi_high = rsi[i]
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 50 for 12h EMA, 30 for volume median, 14 for RSI
    start_idx = max(50, 30, 14) + 10  # extra for divergence lookback
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_median_30[i]) or
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_12h_aligned[i]
        vol_conf = volume_confirm[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Long: bullish divergence + volume confirmation + uptrend (close > EMA50_12h)
            long_entry = bullish_div[i] and vol_conf and (close_val > ema_50_val)
            # Short: bearish divergence + volume confirmation + downtrend (close < EMA50_12h)
            short_entry = bearish_div[i] and vol_conf and (close_val < ema_50_val)
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on trend reversal or RSI overbought (>70)
            if close_val < ema_50_val or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal or RSI oversold (<30)
            if close_val > ema_50_val or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_RSI_Divergence_Volume_HTFTrend"
timeframe = "6h"
leverage = 1.0