#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour momentum with 1-day trend filter and volume confirmation
# Long when 1h close > 1h open (bullish candle), 1d close > 1d open (bullish day), and volume > 1.5x 1h average volume
# Short when 1h close < 1h open (bearish candle), 1d close < 1d open (bearish day), and volume > 1.5x 1h average volume
# Exit when candle direction reverses or trend changes
# Stoploss at 2.0 * ATR(14)
# Position size: 0.20 (20% of capital)
# Uses 1d candle direction for trend filter and 1h volume average for confirmation
# Target: 60-150 total trades over 4 years (15-37/year)

name = "1h_momentum_1d_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    open_price = prices['open'].values
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for trend filter (candle direction)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    open_1d = df_1d['open'].values
    close_1d = df_1d['close'].values
    bullish_1d = close_1d > open_1d  # True for bullish daily candle
    bearish_1d = close_1d < open_1d  # True for bearish daily candle
    bullish_1d_aligned = align_htf_to_ltf(prices, df_1d, bullish_1d.astype(float))
    bearish_1d_aligned = align_htf_to_ltf(prices, df_1d, bearish_1d.astype(float))
    
    # 1-hour candle direction
    bullish_1h = close > open_price
    bearish_1h = close < open_price
    
    # 1-hour volume average for confirmation
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
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
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(volume_ma[i]) or np.isnan(atr[i]) or 
            np.isnan(bullish_1d_aligned[i]) or np.isnan(bearish_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: candle turns bearish or trend turns bearish
            elif bearish_1h[i] or bearish_1d_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: candle turns bullish or trend turns bullish
            elif bullish_1h[i] or bullish_1d_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with candle direction, trend alignment, and volume confirmation
            # Bullish: bullish candle, bullish daily trend, volume spike
            if (bullish_1h[i] and
                bullish_1d_aligned[i] > 0.5 and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: bearish candle, bearish daily trend, volume spike
            elif (bearish_1h[i] and
                  bearish_1d_aligned[i] > 0.5 and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals