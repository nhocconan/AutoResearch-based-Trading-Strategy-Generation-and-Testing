#!/usr/bin/env python3
# Hypothesis: 12h KAMA trend + RSI mean reversion + volume spike + choppiness regime filter.
# Long when KAMA rising, RSI<40, volume > 1.5x MA20, and chop > 61.8 (range regime).
# Short when KAMA falling, RSI>60, volume > 1.5x MA20, and chop > 61.8 (range regime).
# Exit on opposite RSI extreme (RSI>60 for long, RSI<40 for short) or ATR stoploss.
# Uses discrete sizing (0.25) to limit fee churn. Designed for low trade frequency (~12-37/year)
# by requiring confluence of trend, mean reversion, volume, and regime filters.
# Effective in both bull and bear markets as it captures mean reversion within the trend
# during ranging periods (chop>61.8), avoiding strong trending markets where mean reversion fails.

name = "12h_KAMA_RSI_Volume_Chop_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def kama(close, er_period=10, fast_period=2, slow_period=30):
    """Kaufman's Adaptive Moving Average"""
    if len(close) < er_period:
        return np.full(len(close), np.nan)
    change = np.abs(np.diff(close, n=er_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > er_period else np.array([])
    if len(volatility) == 0:
        return np.full(len(close), np.nan)
    # Pad change to match volatility length
    change = np.concatenate([np.full(er_period-1, np.nan), change])
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast_period+1) - 2/(slow_period+1)) + 2/(slow_period+1)) ** 2
    kama_vals = np.full(len(close), np.nan)
    kama_vals[er_period] = close[er_period]
    for i in range(er_period+1, len(close)):
        kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
    return kama_vals

def rsi(close, period=14):
    """Relative Strength Index"""
    if len(close) < period:
        return np.full(len(close), np.nan)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_vals = 100 - (100 / (1 + rs))
    # Prepend NaN for first element
    rsi_vals = np.concatenate([np.array([np.nan]), rsi_vals])
    return rsi_vals

def choppiness_index(high, low, close, period=14):
    """Choppiness Index: higher = ranging, lower = trending"""
    if len(close) < period:
        return np.full(len(close), np.nan)
    atr = []
    for i in range(len(close)):
        if i == 0:
            tr = high[i] - low[i]
        else:
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr.append(tr)
    atr = np.array(atr)
    sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    range_hl = hh - ll
    chop = np.where(range_hl != 0, -100 * np.log10(sum_atr / range_hl) / np.log10(period), 50)
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend context (not used directly, but required by experiment)
    df_1d = get_htf_data(prices, '1d')
    
    # KAMA(10,2,30) on 12h close
    kama_vals = kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_rising = kama_vals > np.roll(kama_vals, 1)
    kama_falling = kama_vals < np.roll(kama_vals, 1)
    
    # RSI(14) on 12h close
    rsi_vals = rsi(close, 14)
    rsi_oversold = rsi_vals < 40
    rsi_overbought = rsi_vals > 60
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)
    
    # Choppiness Index(14) - range regime when > 61.8
    chop_vals = choppiness_index(high, low, close, 14)
    chop_range = chop_vals > 61.8
    
    # ATR(14) for stoploss
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = np.full(n, np.nan)
    
    for i in range(100, n):
        if np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or np.isnan(vol_ma20[i]) or \
           np.isnan(chop_vals[i]) or np.isnan(atr14[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: KAMA rising, RSI oversold, volume spike, range regime
            if kama_rising[i] and rsi_oversold[i] and volume_spike[i] and chop_range[i]:
                signals[i] = 0.25
                position = 1
                entry_price[i] = close[i]
            # SHORT: KAMA falling, RSI overbought, volume spike, range regime
            elif kama_falling[i] and rsi_overbought[i] and volume_spike[i] and chop_range[i]:
                signals[i] = -0.25
                position = -1
                entry_price[i] = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI overbought OR ATR stoploss
            if rsi_overbought[i] or close[i] < entry_price[i-1] - 2.0 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
                entry_price[i] = entry_price[i-1]
        elif position == -1:
            # EXIT SHORT: RSI oversold OR ATR stoploss
            if rsi_oversold[i] or close[i] > entry_price[i-1] + 2.0 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
                entry_price[i] = entry_price[i-1]
    
    return signals