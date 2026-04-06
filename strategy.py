#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d trend filter and volume confirmation
# Enter long when: price breaks above Donchian(20) high, price > 1d EMA(50), volume > 1.5x avg
# Enter short when: price breaks below Donchian(20) low, price < 1d EMA(50), volume > 1.5x avg
# Exit via trailing stop: 3*ATR(14) from extreme
# Targets 75-150 trades over 4 years (19-38/year) with trend-following edge in bull/bear markets

name = "12h_donchian20_1dema_vol_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = high_roll.values
    donchian_low = low_roll.values
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    # ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Update trailing stop
            long_stop = max(long_stop, high[i] - 3.0 * atr[i])
            # Exit: price hits trailing stop OR reverse breakout
            if low[i] <= long_stop or low[i] <= donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Update trailing stop
            short_stop = min(short_stop, low[i] + 3.0 * atr[i])
            # Exit: price hits trailing stop OR reverse breakout
            if high[i] >= short_stop or high[i] >= donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries with trend and volume filters
            if volume[i] > volume_threshold[i]:
                if close[i] > donchian_high[i] and close[i] > ema_50_aligned[i]:
                    # Bullish breakout above Donchian high with 1d uptrend
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    long_stop = high[i] - 3.0 * atr[i]
                elif close[i] < donchian_low[i] and close[i] < ema_50_aligned[i]:
                    # Bearish breakout below Donchian low with 1d downtrend
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    short_stop = low[i] + 3.0 * atr[i]
    
    return signals