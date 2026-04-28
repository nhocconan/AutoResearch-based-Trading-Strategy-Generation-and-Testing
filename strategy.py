#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR(14) stoploss.
# Enter long when price breaks above Donchian upper band (20-bar high) and 1d EMA50 is rising.
# Enter short when price breaks below Donchian lower band (20-bar low) and 1d EMA50 is falling.
# Exit when price touches Donchian middle band (20-bar average of high/low) or ATR-based stoploss.
# Uses discrete position sizing (0.30) to balance return and drawdown.
# Target: 100-200 total trades over 4 years (25-50/year).
# Donchian channels provide clear breakout levels; 1d EMA50 ensures higher timeframe alignment;
# ATR stoploss manages risk in volatile markets. Works in bull (breakouts up) and bear (breakouts down).

name = "4h_Donchian20_1dEMA50_Trend_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 4h
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # EMA50 trend: slope over 3 periods
    ema_50_slope = np.zeros_like(ema_50_aligned)
    ema_50_slope[3:] = (ema_50_aligned[3:] - ema_50_aligned[:-3]) / 3
    ema_trend_up = ema_50_slope > 0
    ema_trend_down = ema_50_slope < 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 14)  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # ATR-based stoploss levels
        if position == 1:  # Long position
            stop_loss = entry_price - 2.0 * atr[i]
        elif position == -1:  # Short position
            stop_loss = entry_price + 2.0 * atr[i]
        else:
            stop_loss = 0.0
        
        price = close[i]
        
        # Handle exits
        if position == 1:  # Long - exit on middle band touch or stoploss
            if price <= donchian_middle[i] or price <= stop_loss:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:  # Short - exit on middle band touch or stoploss
            if price >= donchian_middle[i] or price >= stop_loss:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper band and EMA50 trending up
            if price > donchian_upper[i] and ema_trend_up[i]:
                signals[i] = 0.30
                position = 1
                entry_price = price
            # Short entry: price breaks below Donchian lower band and EMA50 trending down
            elif price < donchian_lower[i] and ema_trend_down[i]:
                signals[i] = -0.30
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
    
    return signals