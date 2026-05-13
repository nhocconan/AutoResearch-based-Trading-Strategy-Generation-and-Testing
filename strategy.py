#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter, volume confirmation (2x MA20), and ATR(14) volatility filter.
# Enters long when price breaks above Camarilla R1 level with 1d bullish trend (close > EMA34), volume > 2x MA20, and ATR(14) > 0.5 * ATR(50).
# Enters short when price breaks below Camarilla S1 level with 1d bearish trend (close < EMA34), volume > 2x MA20, and ATR(14) > 0.5 * ATR(50).
# Exits when price crosses the Camarilla pivot point (PP) or ATR-based stoploss (2.5 * ATR(14) from entry).
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~20-50/year) by requiring strict confluence: price breakout + HTF trend + volume spike + volatility filter.
# Camarilla pivot levels provide high-probability intraday support/resistance derived from prior day's range, effective in both trending and ranging markets.
# The 1d trend filter ensures alignment with higher timeframe direction, while volatility filter avoids low volatility false breakouts.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Volatility_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    range_val = high - low
    pp = (high + low + close) / 3
    r1 = close + range_val * 1.1 / 12
    s1 = close - range_val * 1.1 / 12
    r2 = close + range_val * 1.1 / 6
    s2 = close - range_val * 1.1 / 6
    r3 = close + range_val * 1.1 / 4
    s3 = close - range_val * 1.1 / 4
    r4 = close + range_val * 1.1 / 2
    s4 = close - range_val * 1.1 / 2
    return pp, r1, r2, r3, r4, s1, s2, s3, s4

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
    
    # Calculate Camarilla levels using prior day's OHLC (from 1d data)
    # We need to shift the 1d data by 1 to avoid look-ahead (use prior completed day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_shifted = np.roll(close_1d, 1)
    close_1d_shifted[0] = np.nan  # first value has no prior day
    
    # Calculate Camarilla for each 1d bar using prior day's data
    camarilla_pp = np.full(len(close_1d), np.nan)
    camarilla_r1 = np.full(len(close_1d), np.nan)
    camarilla_s1 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):  # start from 1 to have prior day data
        pp, r1, _, _, _, s1, _, _, _ = calculate_camarilla(
            high_1d[i-1], low_1d[i-1], close_1d[i-1]
        )
        camarilla_pp[i] = pp
        camarilla_r1[i] = r1
        camarilla_s1[i] = s1
    
    # Align Camarilla levels to 4h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume filter: current volume > 2x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)
    
    # ATR(14) and ATR(50) for volatility filter
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr14 > (0.5 * atr50)  # require sufficient volatility
    
    # Track entry price for ATR-based stoploss
    entry_price = np.full(n, np.nan)
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or \
           np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20[i]) or \
           np.isnan(atr14[i]) or np.isnan(atr50[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R1 with 1d bullish trend, volume spike, and sufficient volatility
            if close[i] > camarilla_r1_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i] and volatility_filter[i]:
                signals[i] = 0.25
                position = 1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            # SHORT: Price breaks below Camarilla S1 with 1d bearish trend, volume spike, and sufficient volatility
            elif close[i] < camarilla_s1_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i] and volatility_filter[i]:
                signals[i] = -0.25
                position = -1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Camarilla PP (mean reversion) OR ATR stoploss hit
            if close[i] < camarilla_pp_aligned[i] or close[i] < entry_price[i-1] - 2.5 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
        elif position == -1:
            # EXIT SHORT: Price crosses above Camarilla PP (mean reversion) OR ATR stoploss hit
            if close[i] > camarilla_pp_aligned[i] or close[i] > entry_price[i-1] + 2.5 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
    
    return signals