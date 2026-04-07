#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly trend filter and volume confirmation
# Long when price breaks above 20-day high with volume > 1.5x average and weekly close > weekly open
# Short when price breaks below 20-day low with volume > 1.5x average and weekly close < weekly open
# Exit when price returns to 10-day EMA or volatility expands (ATR ratio > 2.5)
# Stoploss at 2.5 * ATR(20)
# Position size: 0.25 (25% of capital)
# Uses weekly trend to avoid counter-trend trades in ranging markets
# Target: 50-100 total trades over 4 years (12-25/year)

name = "1d_donchian20_weekly_trend_vol_v1"
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
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    weekly_close = df_weekly['close'].values
    weekly_open = df_weekly['open'].values
    
    # Align weekly trend to daily
    weekly_bullish = weekly_close > weekly_open  # True if bullish week
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_weekly, weekly_bullish.astype(float))
    
    # Daily Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 10-day EMA for exit
    close_series = pd.Series(close)
    ema_10 = close_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Volume average (20-period)
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # ATR(20) for volatility and stoploss
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_10[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(atr[i]) or np.isnan(weekly_bullish_aligned[i])):
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
            # Exit: price returns to 10-day EMA or volatility expands significantly
            elif close[i] <= ema_10[i] or (atr[i] > 2.5 * atr[i-1] and i > 0):
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
            # Exit: price returns to 10-day EMA or volatility expands significantly
            elif close[i] >= ema_10[i] or (atr[i] > 2.5 * atr[i-1] and i > 0):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with volume confirmation and weekly trend
            volume_spike = volume[i] > 1.5 * volume_ma[i]
            
            # Long: break above Donchian high with volume spike and weekly bullish
            if (close[i] > donchian_high[i] and 
                volume_spike and 
                weekly_bullish_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: break below Donchian low with volume spike and weekly bearish
            elif (close[i] < donchian_low[i] and 
                  volume_spike and 
                  weekly_bullish_aligned[i] < 0.5):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals