#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h trend filter and volume confirmation
# Long when price breaks above Donchian(20) high, 12h EMA50 uptrend, and volume spike
# Short when price breaks below Donchian(20) low, 12h EMA50 downtrend, and volume spike
# Uses 12h EMA50 for trend filter to avoid counter-trend trades
# Volume confirmation (>1.5x average) to filter false breakouts
# Target: 75-200 total trades over 4 years with controlled risk
# ATR-based stoploss to limit drawdown (2x ATR)

name = "4h_donchian20_12h_ema50_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # EMA50 calculation
    ema50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Donchian(20) channels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if required data not available
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR approximation using price range
            if close[i] < entry_price - 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price falls below Donchian low or trend changes
            elif close[i] < donch_low[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR approximation
            if close[i] > entry_price + 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price rises above Donchian high or trend changes
            elif close[i] > donch_high[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            # Long: price breaks above Donchian high, uptrend, volume spike
            if (close[i] > donch_high[i] and 
                close[i] > ema50_12h_aligned[i] and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low, downtrend, volume spike
            elif (close[i] < donch_low[i] and 
                  close[i] < ema50_12h_aligned[i] and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals