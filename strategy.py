#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend filter + volume confirmation + ATR-based stop
# Long when price breaks above Donchian(20) high AND close > 1d EMA50 AND volume > 1.5x 20-period average
# Short when price breaks below Donchian(20) low AND close < 1d EMA50 AND volume > 1.5x 20-period average
# Exit when price crosses 1d EMA50 (trend filter flip) OR Alligator alignment breaks (if used)
# Uses 4h primary timeframe with 1d HTF for trend filter to capture multi-day moves with controlled frequency
# Discrete sizing (0.30) to balance return and drawdown; targets 75-200 total trades over 4 years (~19-50/year)
# Donchian provides objective breakout levels; EMA50 filters for higher-timeframe trend; volume confirms participation

name = "4h_Donchian20_Breakout_1dEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian Channel (20) on 4h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        bullish_breakout = close[i] > donchian_high[i]
        bearish_breakout = close[i] < donchian_low[i]
        
        if position == 0:
            # Long conditions: bullish breakout AND close > 1d EMA50 AND volume spike
            if bullish_breakout and close[i] > ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.30
                position = 1
            # Short conditions: bearish breakout AND close < 1d EMA50 AND volume spike
            elif bearish_breakout and close[i] < ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: close crosses below 1d EMA50 (trend filter flip)
            if close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: close crosses above 1d EMA50 (trend filter flip)
            if close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals