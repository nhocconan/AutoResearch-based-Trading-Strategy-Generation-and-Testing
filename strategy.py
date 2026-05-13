#!/usr/bin/env python3
# Hypothesis: 6h Williams %R reversal with 1d trend filter and volume confirmation.
# Enters long when Williams %R(14) crosses above -80 from oversold with 1d bullish trend (close > EMA50) and volume > 1.5x MA20.
# Enters short when Williams %R(14) crosses below -20 from overbought with 1d bearish trend (close < EMA50) and volume > 1.5x MA20.
# Exits when Williams %R crosses back through -50 (mean reversion) or ATR-based stoploss (2.0 * ATR14).
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~12-37/year) by requiring confluence: momentum reversal + HTF trend + volume spike.
# Williams %R captures short-term exhaustion, while 1d EMA50 filter ensures alignment with higher timeframe momentum.
# Volume threshold (1.5x) reduces false reversals, improving signal quality in both bull and bear markets.

name = "6h_WilliamsR_Reversal_1dTrend_Volume_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on 1d close
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams %R(14) on 6h timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)
    
    # ATR(14) for stoploss
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Track entry price for ATR-based stoploss
    entry_price = np.full(n, np.nan)
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(williams_r[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20[i]) or np.isnan(atr14[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R crosses above -80 from oversold with 1d bullish trend and volume spike
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            # SHORT: Williams %R crosses below -20 from overbought with 1d bearish trend and volume spike
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses back through -50 (mean reversion) OR ATR stoploss hit
            if (williams_r[i] > -50 and williams_r[i-1] <= -50) or close[i] < entry_price[i-1] - 2.0 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
        elif position == -1:
            # EXIT SHORT: Williams %R crosses back through -50 (mean reversion) OR ATR stoploss hit
            if (williams_r[i] < -50 and williams_r[i-1] >= -50) or close[i] > entry_price[i-1] + 2.0 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
    
    return signals