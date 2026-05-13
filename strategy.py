#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d trend filter (EMA34), volume confirmation (2x MA20), and ATR volatility filter.
# Enters long when price breaks above Camarilla R3 level with 1d bullish trend (close > EMA34), volume > 2x MA20, and ATR(14) > 0.3 * ATR(50).
# Enters short when price breaks below Camarilla S3 level with 1d bearish trend (close < EMA34), volume > 2x MA20, and ATR(14) > 0.3 * ATR(50).
# Exits when price retouches the Camarilla pivot point (PP) or ATR-based stoploss (2 * ATR(14) from entry).
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~19-50/year) by requiring strict confluence: price breakout at strong pivot + HTF trend + volume spike + volatility filter.
# Camarilla pivots identify institutional support/resistance levels; R3/S3 are strong breakout levels.
# The 1d trend filter ensures alignment with higher timeframe direction, while volatility filter avoids low volatility false breakouts.
# This combines proven patterns: Camarilla breakout (top performer) + volume confirmation + volatility filter.

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume_Volatility_v1"
timeframe = "4h"
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
    
    # Calculate Camarilla pivot levels from previous day (using 1d OHLC)
    # Camarilla: PP = (H + L + C) / 3
    # R3 = PP + (H - L) * 1.1 / 2
    # S3 = PP - (H - L) * 1.1 / 2
    # We need previous day's data, so shift by 1
    if len(close_1d) < 2:
        camarilla_pp = np.full(len(close_1d), np.nan)
        camarilla_r3 = np.full(len(close_1d), np.nan)
        camarilla_s3 = np.full(len(close_1d), np.nan)
    else:
        camarilla_pp = (np.roll(df_1d['high'].values, 1) + np.roll(df_1d['low'].values, 1) + np.roll(close_1d, 1)) / 3
        camarilla_r3 = camarilla_pp + (np.roll(df_1d['high'].values, 1) - np.roll(df_1d['low'].values, 1)) * 1.1 / 2
        camarilla_s3 = camarilla_pp - (np.roll(df_1d['high'].values, 1) - np.roll(df_1d['low'].values, 1)) * 1.1 / 2
    # Align to 4h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume filter: current volume > 2.0x 20-period average
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
    volatility_filter = atr14 > (0.3 * atr50)  # avoid low volatility breakouts
    
    # Track entry price for ATR-based stoploss
    entry_price = np.full(n, np.nan)
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or \
           np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20[i]) or \
           np.isnan(atr14[i]) or np.isnan(atr50[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3 with 1d bullish trend, volume spike, and sufficient volatility
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i] and volatility_filter[i]:
                signals[i] = 0.25
                position = 1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            # SHORT: Price breaks below Camarilla S3 with 1d bearish trend, volume spike, and sufficient volatility
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i] and volatility_filter[i]:
                signals[i] = -0.25
                position = -1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price retouches Camarilla PP (mean reversion) OR ATR stoploss hit
            if close[i] < camarilla_pp_aligned[i] or close[i] < entry_price[i-1] - 2.0 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
        elif position == -1:
            # EXIT SHORT: Price retouches Camarilla PP (mean reversion) OR ATR stoploss hit
            if close[i] > camarilla_pp_aligned[i] or close[i] > entry_price[i-1] + 2.0 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
    
    return signals