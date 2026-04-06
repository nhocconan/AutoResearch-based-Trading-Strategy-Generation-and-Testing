#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian(20) breakout with 1-week EMA(50) filter and volume confirmation
# Long when price breaks above 12h Donchian upper band, price > 1w EMA(50), and volume > 1.5x monthly average
# Short when price breaks below 12h Donchian lower band, price < 1w EMA(50), and volume > 1.5x monthly average
# Exit when price returns to 1w EMA(50) or opposite breakout occurs
# Stoploss at 2.5 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 1w EMA for trend filter and monthly volume to confirm breakout strength
# Target: 50-150 total trades over 4 years (12-37/year)

name = "12h_donchian20_1w_ema_1m_vol_v1"
timeframe = "12h"
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
    
    # 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h Donchian(20) channels
    high_series = pd.Series(high_12h)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    low_series = pd.Series(low_12h)
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 12h timeframe (already aligned, but use for safety)
    upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    
    # 1w EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 1m volume for confirmation
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
            np.isnan(ema_1w_aligned[i]) or np.isnan(volume_ma_1m_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to EMA or breaks below lower band
            elif close[i] <= ema_1w_aligned[i] or close[i] < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to EMA or breaks above upper band
            elif close[i] >= ema_1w_aligned[i] or close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: price breaks above upper band, price above EMA (bullish trend), volume spike
            if (close[i] > upper_aligned[i] and
                close[i] > ema_1w_aligned[i] and
                volume[i] > 1.5 * volume_ma_1m_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below lower band, price below EMA (bearish trend), volume spike
            elif (close[i] < lower_aligned[i] and
                  close[i] < ema_1w_aligned[i] and
                  volume[i] > 1.5 * volume_ma_1m_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals