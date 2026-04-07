#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 1-day EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian upper band, EMA50(1d) rising, and volume > 1.5x average
# Short when price breaks below Donchian lower band, EMA50(1d) falling, and volume > 1.5x average
# Exit when price crosses back below Donchian middle band or trend changes
# Stoploss at 2.0 * ATR(14)
# Position size: 0.30 (30% of capital)
# Target: 75-200 total trades over 4 years (19-50/year)

name = "4h_donchian20_1d_ema50_vol_v1"
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
    
    # 4h Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_prev = np.roll(ema50_1d, 1)
    ema50_1d_prev[0] = ema50_1d[0]
    ema50_1d_rising = ema50_1d > ema50_1d_prev
    ema50_1d_falling = ema50_1d < ema50_1d_prev
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema50_1d_rising_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d_rising)
    ema50_1d_falling_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d_falling)
    
    # 4h volume average for confirmation
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
            np.isnan(donchian_mid[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(ema50_1d_rising_aligned[i]) or np.isnan(ema50_1d_falling_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
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
            # Exit: price crosses below Donchian middle or trend turns down
            elif close[i] < donchian_mid[i] or ema50_1d_falling_aligned[i]:
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
            # Exit: price crosses above Donchian middle or trend turns up
            elif close[i] > donchian_mid[i] or ema50_1d_rising_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.30
        else:
            # Look for entries with Donchian breakout, trend alignment, and volume confirmation
            # Bullish breakout: price breaks above Donchian upper band
            bullish_breakout = close[i] > donchian_high[i]
            # Bearish breakout: price breaks below Donchian lower band
            bearish_breakout = close[i] < donchian_low[i]
            
            # Long: bullish breakout, 1d uptrend, volume spike
            if (bullish_breakout and
                ema50_1d_rising_aligned[i] and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.30
                position = 1
                entry_price = close[i]
            # Short: bearish breakout, 1d downtrend, volume spike
            elif (bearish_breakout and
                  ema50_1d_falling_aligned[i] and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.30
                position = -1
                entry_price = close[i]
    
    return signals