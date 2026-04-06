#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d trend filter, volume confirmation, and ATR stoploss
# Enter long when price breaks above Donchian(20) high AND price > 1d EMA(50) AND volume > 1.5x average
# Enter short when price breaks below Donchian(20) low AND price < 1d EMA(50) AND volume > 1.5x average
# Exit via ATR(14) stoploss (2x ATR) or opposite Donchian break
# Targets 75-200 trades over 4 years by requiring confluence of breakout, trend, and volume

name = "4h_donchian20_1dema_vol_atrstop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
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
    
    # ATR(14) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]) or
            np.isnan(atr[i])):
            if position != 0:
                # Maintain position until exit signal
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: stoploss (2*ATR) OR price breaks below Donchian low
            if close[i] <= entry_price - 2.0 * atr[i] or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: stoploss (2*ATR) OR price breaks above Donchian high
            if close[i] >= entry_price + 2.0 * atr[i] or close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian break + trend filter + volume confirmation
            if volume[i] > volume_threshold[i]:
                if close[i] > donchian_high[i] and close[i] > ema_50_aligned[i]:
                    # Bullish breakout above Donchian high with daily uptrend
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif close[i] < donchian_low[i] and close[i] < ema_50_aligned[i]:
                    # Bearish breakout below Donchian low with daily downtrend
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals