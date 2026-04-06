#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d trend filter and volume confirmation
# Long when price breaks above 12h Donchian(20) high AND 1d EMA(50) rising AND volume > 2x average
# Short when price breaks below 12h Donchian(20) low AND 1d EMA(50) falling AND volume > 2x average
# Exit when price crosses 12h EMA(20) in opposite direction
# Uses volatility-adjusted position sizing and ATR stoploss to manage risk
# Designed for low trade frequency (target: 50-150/4 years) to minimize fee drag
# Works in bull markets via trend-following breaks and in bear via short breakdowns

name = "12h_donchian_ema_vol_v2"
timeframe = "12h"
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
    
    # 12h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = highest_high.values
    donchian_low = lowest_low.values
    
    # 12h EMA(20) for exit
    ema_20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # 1d EMA(50) for trend filter (using daily data)
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    ema_50_daily = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_daily)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 2.0 * volume_ma.values
    
    # ATR(14) for stoploss and position sizing
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema_20[i]) or np.isnan(ema_50_aligned[i]) or \
           np.isnan(volume_threshold[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price crosses below EMA(20) OR stoploss hit
            if close[i] < ema_20[i] or close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above EMA(20) OR stoploss hit
            if close[i] > ema_20[i] or close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend filter
            # Long: break above Donchian high + rising 1d EMA(50) + volume spike
            if (close[i] > donchian_high[i-1] and 
                ema_50_aligned[i] > ema_50_aligned[i-1] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: break below Donchian low + falling 1d EMA(50) + volume spike
            elif (close[i] < donchian_low[i-1] and 
                  ema_50_aligned[i] < ema_50_aligned[i-1] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals