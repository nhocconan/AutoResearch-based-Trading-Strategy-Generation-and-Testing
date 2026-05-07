#!/usr/bin/env python3

name = "4h_Bollinger_Band_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Bollinger Bands (20, 2) on 4h data
    period = 20
    std_dev = 2
    sma = np.full(n, np.nan)
    for i in range(period-1, n):
        sma[i] = np.mean(close[i-period+1:i+1])
    variance = np.full(n, np.nan)
    for i in range(period-1, n):
        variance[i] = np.mean((close[i-period+1:i+1] - sma[i]) ** 2)
    std = np.sqrt(variance)
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 1.5x 20-period average (on 4h data)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 6  # Prevent overtrading (approx 1 day for 4h)
    
    start_idx = max(20, 50)  # Warmup for Bollinger Bands and EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine daily trend direction
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
        trend_1d_up = close_1d_aligned[i] > ema_50_1d_aligned[i]
        trend_1d_down = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: price breaks above upper Bollinger Band in daily uptrend with volume confirmation
            if (close[i] > upper_band[i] and 
                trend_1d_up and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: price breaks below lower Bollinger Band in daily downtrend with volume confirmation
            elif (close[i] < lower_band[i] and 
                  trend_1d_down and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: price closes below middle Bollinger Band (SMA) OR trend change
            if (close[i] < sma[i]) or not trend_1d_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price closes above middle Bollinger Band (SMA) OR trend change
            if (close[i] > sma[i]) or not trend_1d_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Bollinger Band breakouts on 4h timeframe, confirmed by daily trend and volume spike.
# Long when price breaks above upper Bollinger Band in daily uptrend with volume spike.
# Short when price breaks below lower Bollinger Band in daily downtrend with volume spike.
# Daily EMA50 filter ensures we trade with the higher timeframe trend.
# Volume confirmation filters out false breakouts. Cooldown prevents overtrading.
# Bollinger Bands adapt to volatility, making them effective in both trending and ranging markets.
# Target: 20-40 trades/year to avoid fee drag. Works in both bull and bear markets by capturing breakouts in direction of daily trend.