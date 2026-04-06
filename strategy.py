#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with volume confirmation and 12-hour trend filter
# Long when price breaks above 4h Donchian upper band, price > 12h EMA(200), and volume > 2x weekly average
# Short when price breaks below 4h Donchian lower band, price < 12h EMA(200), and volume > 2x weekly average
# Exit when price returns to 12h EMA(200) or opposite breakout occurs
# Stoploss at 2.5 * ATR(14) to handle volatility
# Position size: 0.25 (25% of capital) to balance return and drawdown
# Uses 12h EMA for trend filter and weekly volume to confirm breakout strength
# Target: 100-200 total trades over 4 years (25-50/year)

name = "4h_donchian20_12h_ema200_1w_vol_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
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
    
    # Align Donchian bands to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    # 12h EMA(200) for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 200:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=200, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 1w volume for confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    volume_1w = df_1w['volume'].values
    volume_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_1w)
    
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
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(volume_ma_1w_aligned[i]) or 
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
            elif close[i] <= ema_12h_aligned[i] or close[i] < lower_aligned[i]:
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
                volume[i] > 2.0 * volume_ma_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below lower band, price below EMA (bearish trend), volume spike
            elif (close[i] < lower_aligned[i] and
                  close[i] < ema_12h_aligned[i] and
                  volume[i] > 2.0 * volume_ma_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals