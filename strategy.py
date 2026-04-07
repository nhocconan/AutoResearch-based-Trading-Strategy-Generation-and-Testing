#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian channel breakout with daily EMA trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high, daily EMA50 is rising, and volume > 1.5x 12h average volume.
# Short when price breaks below Donchian(20) low, daily EMA50 is falling, and volume > 1.5x 12h average volume.
# Exit when price returns to Donchian midpoint or trend reverses.
# Stoploss at 2.5 * ATR(14).
# Position size: 0.25 (25% of capital).
# Uses daily EMA50 for trend filter and 12h volume average for confirmation.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_donchian20_daily_ema50_vol_v1"
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
    
    # Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Daily EMA slope for trend strength
    ema50_1d_prev = np.roll(ema50_1d_aligned, 1)
    ema50_1d_prev[0] = ema50_1d_aligned[0]
    ema50_slope = ema50_1d_aligned - ema50_1d_prev
    
    # 12h volume average for confirmation
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
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema50_slope[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
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
            # Exit: price returns to Donchian midpoint or trend turns down
            elif close[i] <= donchian_mid[i] or ema50_slope[i] <= 0:
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
            # Exit: price returns to Donchian midpoint or trend turns up
            elif close[i] >= donchian_mid[i] or ema50_slope[i] >= 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with Donchian breakout, trend alignment, and volume confirmation
            # Bullish breakout: price crosses above Donchian high
            bullish_break = close[i] > donchian_high[i] and close[i-1] <= donchian_high[i-1]
            # Bearish breakout: price crosses below Donchian low
            bearish_break = close[i] < donchian_low[i] and close[i-1] >= donchian_low[i-1]
            
            # Long: bullish breakout, daily uptrend (rising EMA), volume spike
            if (bullish_break and
                ema50_slope[i] > 0 and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: bearish breakout, daily downtrend (falling EMA), volume spike
            elif (bearish_break and
                  ema50_slope[i] < 0 and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals