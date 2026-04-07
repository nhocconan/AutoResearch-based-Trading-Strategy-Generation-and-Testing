#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian breakout with weekly trend filter and volume confirmation
# Long when price breaks above Donchian(20) high + weekly close > weekly open (bullish weekly candle) + volume > 1.5x 20-day average volume
# Short when price breaks below Donchian(20) low + weekly close < weekly open (bearish weekly candle) + volume > 1.5x 20-day average volume
# Exit when price crosses Donchian midline (mean reversion) or volume < 0.5x average volume (low conviction)
# Stoploss at 2.0 * ATR(10)
# Position size: 0.30 (30% of capital)
# Uses 1-day Donchian channels for breakouts and weekly trend filter
# Target: 50-100 total trades over 4 years (12-25/year)

name = "1d_donchian20_weekly_trend_volume_v1"
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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly trend: bullish if close > open, bearish if close < open
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open
    
    # Align weekly trend to daily
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    
    # 1-day Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 10-period ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # 20-day average volume for volume filter
    volume_series = pd.Series(volume)
    avg_volume = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(weekly_bullish_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(avg_volume[i])):
            if position != 0:
                signals[i] = position * 0.30
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses Donchian midline (mean reversion) or low volume
            elif close[i] < donchian_mid[i] or volume[i] < 0.5 * avg_volume[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.30
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses Donchian midline (mean reversion) or low volume
            elif close[i] > donchian_mid[i] or volume[i] < 0.5 * avg_volume[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.30
        else:
            # Look for entries: Donchian breakout with weekly trend and volume confirmation
            # Volume filter: volume > 1.5x 20-day average volume (high conviction)
            high_conviction = volume[i] > 1.5 * avg_volume[i]
            
            # Long: break above Donchian high + weekly bullish + high conviction volume
            if close[i] > donchian_high[i] and weekly_bullish_aligned[i] > 0.5 and high_conviction:
                signals[i] = 0.30
                position = 1
                entry_price = close[i]
            # Short: break below Donchian low + weekly bearish + high conviction volume
            elif close[i] < donchian_low[i] and weekly_bullish_aligned[i] < 0.5 and high_conviction:
                signals[i] = -0.30
                position = -1
                entry_price = close[i]
    
    return signals