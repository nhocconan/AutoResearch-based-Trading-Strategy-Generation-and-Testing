#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly trend filter and volume confirmation
# Long: Price breaks above Donchian upper(20) + weekly EMA(20) rising + volume > 1.5x avg
# Short: Price breaks below Donchian lower(20) + weekly EMA(20) falling + volume > 1.5x avg
# Exit: Opposite Donchian break or trailing stop at 2*ATR
# Uses weekly trend to filter breakouts, targeting 50-150 trades over 4 years

name = "6h_donchian20_weekly_ema_vol_v1"
timeframe = "6h"
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
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA(20) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Weekly EMA slope (rising/falling)
    ema_slope = np.diff(ema_20_1w_aligned, prepend=ema_20_1w_aligned[0])
    ema_rising = ema_slope > 0
    ema_falling = ema_slope < 0
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    # ATR(14) for stoploss
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
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(volume_threshold[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price breaks below Donchian lower OR 2*ATR trailing stop
            if close[i] < low_20[i] or close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian upper OR 2*ATR trailing stop
            if close[i] > high_20[i] or close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries: Donchian break + weekly trend + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > high_20[i] and ema_rising[i]:
                    # Bullish breakout with rising weekly trend
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif close[i] < low_20[i] and ema_falling[i]:
                    # Bearish breakout with falling weekly trend
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals