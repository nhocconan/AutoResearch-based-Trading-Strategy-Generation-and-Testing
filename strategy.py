#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian channel breakout with weekly EMA trend filter and volume confirmation
# Long when price breaks above Donchian(20) high, weekly EMA40 shows uptrend, and volume > 1.5x 20-day average
# Short when price breaks below Donchian(20) low, weekly EMA40 shows downtrend, and volume > 1.5x 20-day average
# Exit when price returns to Donchian midpoint or trend reverses
# Stoploss at 2.5 * ATR(20)
# Position size: 0.25 (25% of capital)
# Uses weekly EMA40 for trend filter and daily volume average for confirmation
# Target: 50-100 total trades over 4 years (12-25/year)

name = "1d_donchian20_weekly_ema40_vol_v1"
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
    
    # Daily Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Weekly EMA40 for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 40:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    ema40_weekly = pd.Series(close_weekly).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema40_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema40_weekly)
    
    # Daily volume average for confirmation
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
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
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema40_weekly_aligned[i]) or np.isnan(volume_ma[i]) or 
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
            # Exit: price returns to Donchian midpoint or weekly trend turns down
            elif close[i] <= donchian_mid[i] or close[i] < ema40_weekly_aligned[i]:
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
            # Exit: price returns to Donchian midpoint or weekly trend turns up
            elif close[i] >= donchian_mid[i] or close[i] > ema40_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with Donchian breakout, trend alignment, and volume confirmation
            # Bullish breakout: price closes above Donchian high
            bullish_break = close[i] > donchian_high[i]
            # Bearish breakout: price closes below Donchian low
            bearish_break = close[i] < donchian_low[i]
            
            # Long: bullish breakout, weekly uptrend, volume spike
            if (bullish_break and
                close[i] > ema40_weekly_aligned[i] and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: bearish breakout, weekly downtrend, volume spike
            elif (bearish_break and
                  close[i] < ema40_weekly_aligned[i] and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals