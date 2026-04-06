#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly EMA(10) filter and volume confirmation
# Long when price breaks above daily Donchian upper band, price > weekly EMA(10), and volume > 1.5x daily average
# Short when price breaks below daily Donchian lower band, price < weekly EMA(10), and volume > 1.5x daily average
# Exit when price returns to weekly EMA(10) or opposite breakout occurs
# Stoploss at 2.5 * ATR(20)
# Position size: 0.25 (25% of capital)
# Uses weekly EMA for trend filter and daily volume to confirm breakout strength
# Target: 50-100 total trades over 4 years (12-25/year) for 1d timeframe

name = "1d_donchian20_weekly_ema10_volume_v1"
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
    
    # Daily Donchian(20) channels
    high_series = pd.Series(high)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    low_series = pd.Series(low)
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA(10) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=10, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Daily volume average (20-period) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(20) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(volume_ma[i]) or 
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
            # Exit: price returns to weekly EMA or breaks below lower band
            elif close[i] <= ema_1w_aligned[i] or close[i] < donchian_lower[i]:
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
            # Exit: price returns to weekly EMA or breaks above upper band
            elif close[i] >= ema_1w_aligned[i] or close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: price breaks above upper band, price above weekly EMA (bullish trend), volume spike
            if (close[i] > donchian_upper[i] and
                close[i] > ema_1w_aligned[i] and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below lower band, price below weekly EMA (bearish trend), volume spike
            elif (close[i] < donchian_lower[i] and
                  close[i] < ema_1w_aligned[i] and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals