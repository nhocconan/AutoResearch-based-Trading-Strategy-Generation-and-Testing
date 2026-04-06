#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d trend filter + volume confirmation + ATR stoploss
# Long when price breaks above Donchian(20) high AND 1d EMA50 trend is up AND volume > 1.5x avg
# Short when price breaks below Donchian(20) low AND 1d EMA50 trend is down AND volume > 1.5x avg
# Exit on opposite Donchian break or ATR stoploss (2x ATR)
# Uses 4h timeframe with daily trend filter to capture trends while avoiding counter-trend trades
# Target: 75-200 trades over 4 years (19-50/year)

name = "4h_donchian20_1d_ema50_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = highest_high.values
    donchian_low = lowest_low.values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    ema50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean()
    ema50 = ema50.values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean()
    atr = atr.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema50_aligned[i]) or np.isnan(volume_threshold[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Exit conditions: opposite Donchian break
        if position == 1:  # long position
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian break with trend filter and volume confirmation
            # Long: price breaks above Donchian high AND 1d EMA50 trending up AND volume confirmation
            if (close[i] > donchian_high[i] and ema50_aligned[i] > ema50_aligned[i-1] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low AND 1d EMA50 trending down AND volume confirmation
            elif (close[i] < donchian_low[i] and ema50_aligned[i] < ema50_aligned[i-1] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals