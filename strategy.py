#!/usr/bin/env python3
# Hypothesis: 6h Williams %R reversal with 1d trend filter, volume confirmation, and ATR-based volatility filter.
# Enters long when Williams %R(14) crosses above -80 from oversold with 1d bullish trend (close > EMA34), volume > 1.3x MA20, and ATR14 > 0.2 * ATR50.
# Enters short when Williams %R(14) crosses below -20 from overbought with 1d bearish trend (close < EMA34), volume > 1.3x MA20, and ATR14 > 0.2 * ATR50.
# Exits when Williams %R crosses opposite threshold (-20 for long, -80 for short) or ATR-based stoploss (1.5 * ATR14 from entry).
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~12-37/year) by requiring confluence: Williams %R extreme + HTF trend + volume spike + volatility filter.
# Williams %R identifies exhaustion points, while 1d EMA34 filter ensures alignment with higher timeframe momentum.
# Volume threshold (1.3x) and volatility filter (0.2x) reduce false reversals, improving signal quality in both bull and bear markets.

name = "6h_WilliamsR_Reversal_1dTrend_Volume_Volatility_v1"
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
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Calculate EMA(34) on 1d close
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Williams %R(14) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate Williams %R(14) previous value for crossover detection
    williams_r_prev = np.roll(williams_r, 1)
    williams_r_prev[0] = williams_r[0]  # first bar
    
    # Volume filter: current volume > 1.3x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.3)
    
    # ATR(14) and ATR(50) for volatility filter
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr14 > (0.2 * atr50)  # avoid low volatility reversals
    
    # Track entry price for ATR-based stoploss
    entry_price = np.full(n, np.nan)
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(williams_r_prev[i]) or \
           np.isnan(vol_ma20[i]) or np.isnan(atr14[i]) or np.isnan(atr50[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R crosses above -80 from oversold with 1d bullish trend, volume spike, and sufficient volatility
            if williams_r_prev[i] <= -80 and williams_r[i] > -80 and close[i] > ema34_1d_aligned[i] and volume_spike[i] and volatility_filter[i]:
                signals[i] = 0.25
                position = 1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            # SHORT: Williams %R crosses below -20 from overbought with 1d bearish trend, volume spike, and sufficient volatility
            elif williams_r_prev[i] >= -20 and williams_r[i] < -20 and close[i] < ema34_1d_aligned[i] and volume_spike[i] and volatility_filter[i]:
                signals[i] = -0.25
                position = -1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses above -20 (overbought) OR ATR stoploss hit
            if williams_r_prev[i] >= -20 and williams_r[i] < -20 or close[i] < entry_price[i-1] - 1.5 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
        elif position == -1:
            # EXIT SHORT: Williams %R crosses below -80 (oversold) OR ATR stoploss hit
            if williams_r_prev[i] <= -80 and williams_r[i] > -80 or close[i] > entry_price[i-1] + 1.5 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
    
    return signals