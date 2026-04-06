# 1d Donchian Breakout with Weekly EMA Filter and Volume Confirmation
# Hypothesis: Daily Donchian(20) breakouts aligned with weekly EMA(50) trend and volume confirmation work in both bull and bear markets.
# The weekly EMA acts as a trend filter to avoid counter-trend trades, while volume confirms breakout strength.
# Stoploss at 2x ATR(14) limits downside. Target: 30-100 trades over 4 years.
# Weekly timeframe provides clean trend signal without whipsaw; daily breakouts capture momentum.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_weekly_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Daily Donchian(20) channels
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # ATR(14) for volatility and stoploss
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.mean(tr[i-14:i+1])
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(ema_aligned[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            stop_loss = entry_price - 2.0 * atr[i]
            # Exit: price breaks below Donchian low or stoploss
            if close[i] <= donch_low[i] or close[i] < stop_loss:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            stop_loss = entry_price + 2.0 * atr[i]
            # Exit: price breaks above Donchian high or stoploss
            if close[i] >= donch_high[i] or close[i] > stop_loss:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend and volume filters
            if volume_filter:
                # Long: price breaks above Donchian high AND price above weekly EMA
                if close[i] > donch_high[i] and close[i] > ema_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: price breaks below Donchian low AND price below weekly EMA
                elif close[i] < donch_low[i] and close[i] < ema_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals