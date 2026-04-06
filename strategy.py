#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 12-hour EMA(30) filter and volume confirmation
# Long when price breaks above 4h Donchian upper band, price > 12h EMA(30), and volume > 2.0x monthly average
# Short when price breaks below 4h Donchian lower band, price < 12h EMA(30), and volume > 2.0x monthly average
# Exit when price returns to 12h EMA(30) or opposite breakout occurs
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 12h EMA for trend filter and monthly volume to confirm breakout strength
# Target: 75-200 total trades over 4 years (19-50/year)

name = "4h_donchian20_12h_ema_1m_vol_v1"
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
    
    # 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h Donchian(20) channels
    high_series = pd.Series(high_4h)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    low_series = pd.Series(low_4h)
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 4h timeframe (already aligned, but use for safety)
    upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    # 12h EMA(30) for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=30, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 1m volume for confirmation (monthly = ~4 weeks)
    df_1m = get_htf_data(prices, '1m')
    if len(df_1m) < 20:
        return np.zeros(n)
    
    volume_1m = df_1m['volume'].values
    volume_ma_1m = pd.Series(volume_1m).rolling(window=20, min_periods=20).mean().values
    volume_ma_1m_aligned = align_htf_to_ltf(prices, df_1m, volume_ma_1m)
    
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
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(volume_ma_1m_aligned[i]) or 
            np.isnan(atr[i])):
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
            # Exit: price returns to EMA or breaks below lower band
            elif close[i] <= ema_12h_aligned[i] or close[i] < lower_aligned[i]:
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
            # Exit: price returns to EMA or breaks above upper band
            elif close[i] >= ema_12h_aligned[i] or close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: price breaks above upper band, price above EMA (bullish trend), volume spike
            if (close[i] > upper_aligned[i] and
                close[i] > ema_12h_aligned[i] and
                volume[i] > 2.0 * volume_ma_1m_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below lower band, price below EMA (bearish trend), volume spike
            elif (close[i] < lower_aligned[i] and
                  close[i] < ema_12h_aligned[i] and
                  volume[i] > 2.0 * volume_ma_1m_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals