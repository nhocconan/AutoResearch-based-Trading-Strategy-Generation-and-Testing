#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly trend filter and volume confirmation
# Long when price breaks above daily Donchian upper band, weekly close > weekly EMA10 (uptrend), and volume > 2x daily average volume
# Short when price breaks below daily Donchian lower band, weekly close < weekly EMA10 (downtrend), and volume > 2x daily average volume
# Exit when trend reverses (weekly close crosses EMA10) or opposite breakout occurs
# Stoploss at 2.5 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses weekly EMA10 for trend filter and daily volume average for confirmation
# Target: 40-100 total trades over 4 years (10-25/year)

name = "1d_donchian20_weekly_ema10_volume_v1"
timeframe = "1d"
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
    
    # Daily Donchian(20) channels
    high_series = pd.Series(high)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    low_series = pd.Series(low)
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Weekly data for EMA10 trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    ema_weekly = pd.Series(close_weekly).ewm(span=10, adjust=False).mean().values
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Daily volume average for confirmation
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
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
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_weekly_aligned[i]) or np.isnan(volume_ma[i]) or 
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
            # Exit: trend reverses (weekly close below EMA10) or breaks below lower band
            elif close[i] < ema_weekly_aligned[i] or close[i] < donchian_lower[i]:
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
            # Exit: trend reverses (weekly close above EMA10) or breaks above upper band
            elif close[i] > ema_weekly_aligned[i] or close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: price breaks above upper band, weekly close above EMA10 (uptrend), volume spike
            if (close[i] > donchian_upper[i] and
                close[i] > ema_weekly_aligned[i] and
                volume[i] > 2.0 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below lower band, weekly close below EMA10 (downtrend), volume spike
            elif (close[i] < donchian_lower[i] and
                  close[i] < ema_weekly_aligned[i] and
                  volume[i] > 2.0 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals