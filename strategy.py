#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian breakout with 4h trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper (20) + 4h EMA(50) up + volume > 1.5x avg
# Short when price breaks below 4h Donchian lower (20) + 4h EMA(50) down + volume > 1.5x avg
# Exit when price crosses 4h EMA(50) or ATR-based stop (2*ATR)
# Uses 4h for signal direction to reduce trade frequency, targets 100-150 total trades over 4 years
# Works in bull (breakouts up) and bear (breakouts down) by following 4h trend

name = "1h_donchian_4h_trend_vol_v1"
timeframe = "1h"
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
    
    # 4h data for trend and Donchian channels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h EMA(50) for trend filter
    close_4h_series = pd.Series(close_4h)
    ema_50_4h = close_4h_series.ewm(span=50, adjust=False).mean().values
    
    # 4h Donchian channels (20-period)
    high_4h_series = pd.Series(high_4h)
    low_4h_series = pd.Series(low_4h)
    donch_high_20 = high_4h_series.rolling(window=20, min_periods=20).max().values
    donch_low_20 = low_4h_series.rolling(window=20, min_periods=20).min().values
    
    # Align 4h indicators to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    donch_high_20_aligned = align_htf_to_ltf(prices, df_4h, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_4h, donch_low_20)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    # ATR for dynamic stop (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(donch_high_20_aligned[i]) or \
           np.isnan(donch_low_20_aligned[i]) or np.isnan(volume_threshold[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            # Exit: price crosses below 4h EMA(50) OR 2*ATR stop loss
            if close[i] < ema_50_4h_aligned[i] or close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price crosses above 4h EMA(50) OR 2*ATR stop loss
            if close[i] > ema_50_4h_aligned[i] or close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: Donchian breakout with 4h trend filter and volume confirmation
            # Long: price breaks above 4h Donchian upper + 4h EMA(50) rising + volume confirmation
            if (close[i] > donch_high_20_aligned[i] and 
                ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: price breaks below 4h Donchian lower + 4h EMA(50) falling + volume confirmation
            elif (close[i] < donch_low_20_aligned[i] and 
                  ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals