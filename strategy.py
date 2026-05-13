#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and ATR-based volatility regime.
# Long when price breaks above Donchian upper (20) AND close > 1w EMA50 AND ATR14 > ATR50 (high vol).
# Short when price breaks below Donchian lower (20) AND close < 1w EMA50 AND ATR14 > ATR50.
# Exit when price returns to Donchian middle (10-period average of upper/lower) OR ATR14 < ATR50 * 0.8.
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~12-37/year) by requiring confluence of breakout, weekly trend, and volatility regime.
# Donchian channels provide clear breakout levels that work in both bull and bear markets.
# Weekly EMA50 ensures we only trade in the direction of the higher timeframe trend.
# ATR volatility regime filters out low-volatility choppy periods where breakouts fail.

name = "12h_Donchian20_Breakout_1wTrend_VolRegime_v1"
timeframe = "12h"
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
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on 1w close for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (highest_high + lowest_low) / 2
    
    # ATR(14) and ATR(50) for volatility regime
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr3 = np.absolute(close - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]  # first bar
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Volatility regime: ATR14 > ATR50 (high volatility)
    vol_regime = atr14 > atr50
    
    # Track entry price for stoploss (optional, using signal reversal as primary exit)
    entry_price = np.full(n, np.nan)
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or \
           np.isnan(donchian_middle[i]) or np.isnan(atr14[i]) or np.isnan(atr50[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper AND close > 1w EMA50 AND high volatility regime
            if close[i] > highest_high[i] and close[i] > ema50_1w_aligned[i] and vol_regime[i]:
                signals[i] = 0.25
                position = 1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            # SHORT: Price breaks below Donchian lower AND close < 1w EMA50 AND high volatility regime
            elif close[i] < lowest_low[i] and close[i] < ema50_1w_aligned[i] and vol_regime[i]:
                signals[i] = -0.25
                position = -1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to Donchian middle OR low volatility regime (ATR14 < ATR50 * 0.8)
            if close[i] <= donchian_middle[i] or atr14[i] < atr50[i] * 0.8:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
        elif position == -1:
            # EXIT SHORT: Price returns to Donchian middle OR low volatility regime (ATR14 < ATR50 * 0.8)
            if close[i] >= donchian_middle[i] or atr14[i] < atr50[i] * 0.8:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
    
    return signals