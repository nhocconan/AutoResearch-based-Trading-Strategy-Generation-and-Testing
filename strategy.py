#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with 1-week EMA20 trend filter and volume confirmation
# Long when price breaks above Donchian(20) high + close > EMA20(1w) + volume > 1.8x average
# Short when price breaks below Donchian(20) low + close < EMA20(1w) + volume > 1.8x average
# Uses 1-week EMA20 for trend filter to avoid counter-trend trades in bear markets
# Target: 30-100 total trades over 4 years with controlled risk
# ATR-based stoploss to limit drawdown (2.5x ATR)

name = "1d_donchian20_1w_ema20_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # EMA20 calculation on weekly data
    ema20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Align 1w EMA20 to daily timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Donchian channels (20-period daily)
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Volume average (20-period daily)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR calculation (14-period) for stoploss
    def calculate_atr(high, low, close, period=14):
        tr1 = pd.Series(high).rolling(window=1).max() - pd.Series(low).rolling(window=1).min()
        tr2 = abs(pd.Series(high).rolling(window=1).max() - pd.Series(close).shift(1))
        tr3 = abs(pd.Series(low).rolling(window=1).min() - pd.Series(close).shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=period).mean().values
        return atr
    
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    max_price_since_entry = 0.0
    min_price_since_entry = 0.0
    
    for i in range(60, n):
        # Skip if required data not available
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            max_price_since_entry = max(max_price_since_entry, high[i])
            # Stoploss: 2.5 * ATR
            if close[i] < max_price_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                max_price_since_entry = 0.0
            # Exit: price breaks below Donchian lower or trend changes
            elif close[i] < donchian_lower[i] or close[i] < ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                max_price_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            min_price_since_entry = min(min_price_since_entry, low[i])
            # Stoploss: 2.5 * ATR
            if close[i] > min_price_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                min_price_since_entry = 0.0
            # Exit: price breaks above Donchian upper or trend changes
            elif close[i] > donchian_upper[i] or close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                min_price_since_entry = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            # Long: break above Donchian upper + uptrend + volume spike
            if (close[i] > donchian_upper[i] and 
                close[i] > ema20_1w_aligned[i] and
                volume[i] > 1.8 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                max_price_since_entry = high[i]
            # Short: break below Donchian lower + downtrend + volume spike
            elif (close[i] < donchian_lower[i] and 
                  close[i] < ema20_1w_aligned[i] and
                  volume[i] > 1.8 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                min_price_since_entry = low[i]
    
    return signals