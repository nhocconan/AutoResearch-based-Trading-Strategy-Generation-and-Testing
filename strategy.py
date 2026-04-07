#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian(20) breakout with 1-week volume confirmation and 1-month ATR volatility filter
# Long when price breaks above 20-day Donchian high + weekly volume > 1.5x 4-week average + monthly ATR < 0.03 * price
# Short when price breaks below 20-day Donchian low + weekly volume > 1.5x 4-week average + monthly ATR < 0.03 * price
# Exit when price crosses 10-day EMA in opposite direction
# Stoploss at 2.0 * ATR(10)
# Position size: 0.25 (25% of capital)
# Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe

name = "1d_donchian20_1w_vol_1m_atr_v1"
timeframe = "1d"
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
    
    # 1-week data for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1-month data for ATR volatility filter
    df_1m = get_htf_data(prices, '1m')
    if len(df_1m) < 10:
        return np.zeros(n)
    
    # 1-week volume average (4-week)
    volume_1w = df_1w['volume'].values
    volume_1w_s = pd.Series(volume_1w)
    volume_ma = volume_1w_s.rolling(window=4, min_periods=4).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1w, volume_ma)
    
    # 1-month ATR(10) for volatility filter
    high_1m = df_1m['high'].values
    low_1m = df_1m['low'].values
    close_1m = df_1m['close'].values
    
    tr1 = high_1m - low_1m
    tr2 = np.abs(high_1m - np.roll(close_1m, 1))
    tr3 = np.abs(low_1m - np.roll(close_1m, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_1m = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1m = pd.Series(tr_1m).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr_1m_aligned = align_htf_to_ltf(prices, df_1m, atr_1m)
    
    # 20-day Donchian channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 10-day EMA for exit
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # ATR(10) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_1m_aligned[i]) or 
            np.isnan(ema_10[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below 10-day EMA
            elif close[i] < ema_10[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above 10-day EMA
            elif close[i] > ema_10[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with volume confirmation and volatility filter
            # Volume filter: weekly volume > 1.5x 4-week average
            volume_filter = volume_ma_aligned[i] > 0 and volume[i] > 1.5 * volume_ma_aligned[i]
            # Volatility filter: monthly ATR < 3% of price (avoid extremely volatile periods)
            vol_filter = atr_1m_aligned[i] < 0.03 * close[i]
            
            # Long: price breaks above Donchian high + volume filter + vol filter
            if close[i] > highest_high[i] and volume_filter and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low + volume filter + vol filter
            elif close[i] < lowest_low[i] and volume_filter and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals