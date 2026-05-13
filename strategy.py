#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 1w EMA50 trend filter and ATR-based volatility regime.
# Elder Ray: Bull Power = High - EMA13(close), Bear Power = Low - EMA13(close).
# Long when Bull Power > 0 AND Bear Power < 0 (bullish bias) AND price > 1w EMA50 AND ATR14 > ATR50 (high vol).
# Short when Bull Power < 0 AND Bear Power > 0 (bearish bias) AND price < 1w EMA50 AND ATR14 > ATR50.
# Exit when Elder Ray bias reverses OR ATR14 < ATR50 * 0.8 (low vol exit).
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~12-37/year) by requiring confluence of Elder Ray bias, weekly trend, and volatility regime.
# Elder Ray measures bull/bear power relative to trend (EMA13), effective in both bull and bear markets by filtering with weekly trend and volatility.

name = "6h_ElderRay_BullBearPower_1wTrend_VolRegime_v1"
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
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on 1w close for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Elder Ray: EMA13 of close
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # ATR(14) and ATR(50) for volatility regime
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
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
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or \
           np.isnan(atr14[i]) or np.isnan(atr50[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 AND Bear Power < 0 (bullish bias), price > 1w EMA50, high volatility regime
            if bull_power[i] > 0 and bear_power[i] < 0 and close[i] > ema50_1w_aligned[i] and vol_regime[i]:
                signals[i] = 0.25
                position = 1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            # SHORT: Bull Power < 0 AND Bear Power > 0 (bearish bias), price < 1w EMA50, high volatility regime
            elif bull_power[i] < 0 and bear_power[i] > 0 and close[i] < ema50_1w_aligned[i] and vol_regime[i]:
                signals[i] = -0.25
                position = -1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Elder Ray bias reverses (Bull Power <= 0 OR Bear Power >= 0) OR low volatility regime (ATR14 < ATR50 * 0.8)
            if bull_power[i] <= 0 or bear_power[i] >= 0 or atr14[i] < atr50[i] * 0.8:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
        elif position == -1:
            # EXIT SHORT: Elder Ray bias reverses (Bull Power >= 0 OR Bear Power <= 0) OR low volatility regime (ATR14 < ATR50 * 0.8)
            if bull_power[i] >= 0 or bear_power[i] <= 0 or atr14[i] < atr50[i] * 0.8:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
    
    return signals