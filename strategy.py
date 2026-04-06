#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA(50) trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high, price > 1d EMA50, and volume > 1.3x 20-period average.
# Short when price breaks below Donchian(20) low, price < 1d EMA50, and volume > 1.3x 20-period average.
# Uses ATR-based stoploss (2.5 * ATR) to limit drawdown.
# Target: 75-150 total trades over 4 years (19-38/year) to avoid fee drag.

name = "4h_donchian20_1d_ema_vol_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=20, min_periods=20).max().values
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 1d trend filter: 50-period EMA on daily chart
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50d_aligned = align_htf_to_ltf(prices, df_1d, ema_50d)
    
    # Volume filter: current volume > 1.3x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if 1d trend data not available
        if np.isnan(ema_50d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below Donchian low or stoploss
            stop_loss_level = entry_price - 2.5 * atr[i]
            
            if (close[i] < donch_low[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high or stoploss
            stop_loss_level = entry_price + 2.5 * atr[i]
            
            if (close[i] > donch_high[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and 1d trend filter
            if volume_filter:
                # Long: price breaks above Donchian high and price > 1d EMA50
                if (close[i] > donch_high[i] and 
                    close[i] > ema_50d_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: price breaks below Donchian low and price < 1d EMA50
                elif (close[i] < donch_low[i] and 
                      close[i] < ema_50d_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals