#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR(14) volatility filter.
- Primary timeframe: 4h for lower trade frequency (target 20-50 trades/year per symbol).
- HTF: 1d EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Volatility filter: Only trade when current ATR(14) > 0.5 * 20-period ATR MA to avoid low-volatility chop.
- Entry: Long when price breaks above Donchian high(20) AND 1d EMA50 bullish AND volatility filter pass.
         Short when price breaks below Donchian low(20) AND 1d EMA50 bearish AND volatility filter pass.
- Exit: Opposite Donchian breakout (long exits on low break, short exits on high break) or loss of volatility confirmation.
- Signal size: 0.30 discrete to balance return and drawdown while minimizing fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
This strategy captures institutional breakouts in the direction of the daily trend, filtered by sufficient volatility
to avoid false signals in ranging markets. Works in both bull and bear markets by only taking trend-aligned trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d ATR(14) and its 20-period MA for volatility filter
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # True Range calculation
    tr1 = df_1d_high[1:] - df_1d_low[1:]
    tr2 = np.abs(df_1d_high[1:] - df_1d_close[:-1])
    tr3 = np.abs(df_1d_low[1:] - df_1d_close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with original indices
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period) on 4h data
    # Donchian high = max(high, lookback=20)
    # Donchian low = min(low, lookback=20)
    high_rolling_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align HTF indicators to 4h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    # Volatility filter: current ATR > 0.5 * 20-period ATR MA
    volatility_filter = atr_1d_aligned > (0.5 * atr_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # Need enough bars for EMA50, Donchian20, ATR14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(volatility_filter[i]) or 
            np.isnan(high_rolling_max[i]) or np.isnan(low_rolling_min[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for breakout signals with volatility filter
            if volatility_filter[i]:
                # Bullish breakout: price > Donchian high(20) AND 1d EMA50 bullish (close > EMA)
                if curr_high > high_rolling_max[i] and curr_close > ema_1d_aligned[i]:
                    signals[i] = 0.30
                    position = 1
                # Bearish breakout: price < Donchian low(20) AND 1d EMA50 bearish (close < EMA)
                elif curr_low < low_rolling_min[i] and curr_close < ema_1d_aligned[i]:
                    signals[i] = -0.30
                    position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low(20) OR loss of volatility confirmation
            if curr_low < low_rolling_min[i] or not volatility_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price breaks above Donchian high(20) OR loss of volatility confirmation
            if curr_high > high_rolling_max[i] or not volatility_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_1dEMA50_Trend_ATRVolatilityFilter_v1"
timeframe = "4h"
leverage = 1.0