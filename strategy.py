#!/usr/bin/env python3
# Hypothesis: 12h Williams %R with 1d EMA34 trend filter and ATR-based volatility regime.
# Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100 over 14 periods.
# Long when %R crosses above -80 (oversold bounce) AND price > 1d EMA34 AND ATR14 > ATR50 (high vol).
# Short when %R crosses below -20 (overbought rejection) AND price < 1d EMA34 AND ATR14 > ATR50.
# Exit when %R crosses below -50 (for long) or above -50 (for short) OR ATR14 < ATR50 * 0.8 (low vol exit).
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~12-37/year) by requiring confluence of %R extreme, daily trend, and volatility regime.
# Williams %R identifies overbought/oversold conditions; effective in ranging markets with trend filter avoids false signals in strong trends.

name = "12h_Williams_%R_1dTrend_VolRegime_v1"
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Williams %R (14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    williams_r[highest_high == lowest_low] = -50  # avoid division by zero
    
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
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(williams_r[i]) or \
           np.isnan(atr14[i]) or np.isnan(atr50[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R crosses above -80 (from below), price > 1d EMA34, high volatility regime
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and close[i] > ema34_1d_aligned[i] and vol_regime[i]:
                signals[i] = 0.25
                position = 1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            # SHORT: Williams %R crosses below -20 (from above), price < 1d EMA34, high volatility regime
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and close[i] < ema34_1d_aligned[i] and vol_regime[i]:
                signals[i] = -0.25
                position = -1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses below -50 (from above) OR low volatility regime (ATR14 < ATR50 * 0.8)
            if williams_r[i] < -50 and williams_r[i-1] >= -50 or atr14[i] < atr50[i] * 0.8:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
        elif position == -1:
            # EXIT SHORT: Williams %R crosses above -50 (from below) OR low volatility regime (ATR14 < ATR50 * 0.8)
            if williams_r[i] > -50 and williams_r[i-1] <= -50 or atr14[i] < atr50[i] * 0.8:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
    
    return signals