#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(15) breakout with 12h trend filter (EMA25) and volume confirmation
# Uses 12h EMA25 as trend filter to avoid counter-trend trades
# Long when price breaks above Donchian(15) high + close > EMA25 + volume > 1.8x average
# Short when price breaks below Donchian(15) low + close < EMA25 + volume > 1.8x average
# Target: 75-150 total trades over 4 years with controlled risk
# ATR-based stoploss (2.5x ATR) to limit drawdown

name = "6h_donchian15_12h_ema25_vol_v1"
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
    
    # 12h data for EMA25 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 25:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # EMA25 calculation
    ema25_12h = pd.Series(close_12h).ewm(span=25, min_periods=25, adjust=False).mean().values
    
    # Align 12h EMA25 to 6h timeframe
    ema25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema25_12h)
    
    # Donchian channels (15-period)
    def calculate_donchian(high, low, period=15):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 15)
    
    # Volume average (15-period)
    volume_ma = pd.Series(volume).rolling(window=15, min_periods=15).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema25_12h_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR approximation using price range
            if close[i] < entry_price - 2.5 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Donchian lower or trend changes
            elif close[i] < donchian_lower[i] or close[i] < ema25_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR approximation
            if close[i] > entry_price + 2.5 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Donchian upper or trend changes
            elif close[i] > donchian_upper[i] or close[i] > ema25_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            # Long: break above Donchian upper + uptrend + volume spike
            if (close[i] > donchian_upper[i] and 
                close[i] > ema25_12h_aligned[i] and
                volume[i] > 1.8 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: break below Donchian lower + downtrend + volume spike
            elif (close[i] < donchian_lower[i] and 
                  close[i] < ema25_12h_aligned[i] and
                  volume[i] > 1.8 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals