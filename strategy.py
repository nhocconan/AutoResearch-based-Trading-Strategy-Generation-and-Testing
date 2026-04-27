#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 12h trend filter and volume confirmation
# Uses 12h EMA50 for trend direction (long when price > EMA50, short when price < EMA50)
# and 4h Donchian(20) breakouts for entry. Volume > 1.8x 20-period average confirms breakout.
# The 12h trend filter reduces whipsaws by ensuring alignment with intermediate-term trend.
# Donchian channels provide clear breakout levels that work in both trending and ranging markets.
# Target: 25-35 trades/year to minimize fee decay while capturing high-probability moves.
# Focus on BTC/ETH as primary assets.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 4h Donchian channel (20-period)
    donchian_period = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(donchian_period, n):
        upper[i] = np.max(high[i-donchian_period:i])
        lower[i] = np.min(low[i-donchian_period:i])
    
    # 20-period average volume for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(donchian_period, vol_period, 1)
    
    for i in range(start_idx, n):
        if (np.isnan(upper[i]) or 
            np.isnan(lower[i]) or
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Determine trend from 12h EMA50
        uptrend = price > ema_50_12h_aligned[i]
        downtrend = price < ema_50_12h_aligned[i]
        
        # Breakout conditions at Donchian levels
        breakout_up = price > upper[i]
        breakdown_down = price < lower[i]
        
        # Volume confirmation: spike > 1.8x average
        volume_confirmation = vol_ratio > 1.8
        
        if position == 0:
            # Long: bullish breakout above upper band with uptrend and volume
            if uptrend and breakout_up and volume_confirmation:
                signals[i] = size
                position = 1
            # Short: bearish breakdown below lower band with downtrend and volume
            elif downtrend and breakdown_down and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns to lower band or breaks below 12h EMA50
            if price < lower[i] or price < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price returns to upper band or breaks above 12h EMA50
            if price > upper[i] or price > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_20_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0