#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly EMA200 trend filter and volume confirmation
# Long when price breaks above daily Donchian upper band, weekly close > weekly EMA200 (uptrend), and daily volume > 1.5x 5-day average volume
# Short when price breaks below daily Donchian lower band, weekly close < weekly EMA200 (downtrend), and daily volume > 1.5x 5-day average volume
# Exit when trend reverses (weekly close crosses EMA200) or opposite breakout occurs
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses weekly EMA200 for trend filter and daily volume average for confirmation
# Target: 50-100 total trades over 4 years (12-25/year)

name = "1d_donchian20_weekly_ema200_vol_v1"
timeframe = "1d"
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
    
    # Daily data for Donchian channels
    df_daily = prices.copy()
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    
    # Daily Donchian(20) channels
    high_series = pd.Series(high_daily)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    low_series = pd.Series(low_daily)
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Weekly data for EMA200 trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 200:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    ema_weekly = pd.Series(close_weekly).ewm(span=200, adjust=False).mean().values
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Daily volume average for confirmation (5-day)
    volume_series = pd.Series(volume)
    volume_ma_daily = volume_series.rolling(window=5, min_periods=5).mean().values
    
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
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_weekly_aligned[i]) or np.isnan(volume_ma_daily[i]) or 
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
            # Exit: trend reverses (price below EMA200) or breaks below lower band
            elif close[i] < ema_weekly_aligned[i] or close[i] < donchian_lower[i]:
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
            # Exit: trend reverses (price above EMA200) or breaks above upper band
            elif close[i] > ema_weekly_aligned[i] or close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: price breaks above upper band, price above EMA200 (uptrend), volume spike
            if (close[i] > donchian_upper[i] and
                close[i] > ema_weekly_aligned[i] and
                volume[i] > 1.5 * volume_ma_daily[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below lower band, price below EMA200 (downtrend), volume spike
            elif (close[i] < donchian_lower[i] and
                  close[i] < ema_weekly_aligned[i] and
                  volume[i] > 1.5 * volume_ma_daily[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals