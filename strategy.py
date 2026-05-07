#!/usr/bin/env python3
name = "4h_KAMA_RSI_Chop_Filter"
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
    
    # Get 1d data for chop filter (choppy market detection)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Chop filter: 14-period chop index (0-100) - high = choppy, low = trending
    atr_14_1d = np.zeros(len(df_1d))
    tr_1d = np.maximum(df_1d['high'].values - df_1d['low'].values,
                       np.maximum(np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1)),
                                  np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))))
    tr_1d[0] = df_1d['high'].values[0] - df_1d['low'].values[0]
    for i in range(1, len(tr_1d)):
        atr_14_1d[i] = (atr_14_1d[i-1] * 13 + tr_1d[i]) / 14 if i >= 1 else tr_1d[i]
    
    # Chop index: 100 * log10(sum(atr14) / (max(high) - min(low))) / log10(14)
    chop_raw = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        sum_atr = np.sum(atr_14_1d[i-13:i+1])
        highest_high = np.max(df_1d['high'].values[i-13:i+1])
        lowest_low = np.min(df_1d['low'].values[i-13:i+1])
        range_hl = highest_high - lowest_low
        if range_hl > 0:
            chop_raw[i] = 100 * np.log10(sum_atr / range_hl) / np.log10(14)
        else:
            chop_raw[i] = 50  # neutral when no range
    
    chop_filter = align_htf_to_ltf(prices, df_1d, chop_raw)
    chop_threshold = 61.8  # >61.8 = choppy/range (mean reversion), <38.2 = trending
    
    # KAMA on 4h close for trend direction
    # Efficiency Ratio: abs(close - close[10]) / sum(abs(diff)) over 10 periods
    change = np.abs(np.subtract(close[9:], close[:-9]))  # close[i] - close[i-9]
    volatility = np.sum(np.abs(np.diff(close.reshape(-1, 1))[:9]), axis=1) if len(close) >= 10 else np.array([])
    er = np.full(n, np.nan)
    if len(change) == len(volatility) and len(change) > 0:
        er[9:] = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = np.square(er * (fast_sc - slow_sc) + slow_sc)
    
    # KAMA calculation
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI on 4h for overbought/oversold
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[13] = np.mean(gain[1:14]) if len(gain) >= 14 else 0
    avg_loss[13] = np.mean(loss[1:14]) if len(loss) >= 14 else 0
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 2  # ~8 hours for 4h
    
    start_idx = max(30, 20, 14, 10)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop_filter[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Market regime: choppy (>61.8) = mean reversion, trending (<38.2) = trend follow
        is_choppy = chop_filter[i] > chop_threshold
        is_trending = chop_filter[i] < (100 - chop_threshold)  # < 38.2
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            if is_choppy:
                # Mean reversion in chop: buy oversold, sell overbought
                if rsi[i] < 30 and kama[i] > close[i] and vol_filter[i]:  # Oversold but KAMA above price = bullish bias
                    signals[i] = 0.25
                    position = 1
                    bars_since_last_trade = 0
                elif rsi[i] > 70 and kama[i] < close[i] and vol_filter[i]:  # Overbought but KAMA below price = bearish bias
                    signals[i] = -0.25
                    position = -1
                    bars_since_last_trade = 0
            elif is_trending:
                # Trend following: buy when price > KAMA and bullish, sell when price < KAMA and bearish
                if close[i] > kama[i] and rsi[i] > 50 and rsi[i] < 70 and vol_filter[i]:  # Bullish but not overbought
                    signals[i] = 0.25
                    position = 1
                    bars_since_last_trade = 0
                elif close[i] < kama[i] and rsi[i] < 50 and rsi[i] > 30 and vol_filter[i]:  # Bearish but not oversold
                    signals[i] = -0.25
                    position = -1
                    bars_since_last_trade = 0
        elif position == 1:
            # Exit long: price crosses below KAMA or RSI overbought
            if close[i] < kama[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above KAMA or RSI oversold
            if close[i] > kama[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: KAMA trend + RSI momentum + Chop regime filter on 4h timeframe.
# In choppy markets (CHOP > 61.8): mean reversion - buy RSI<30 when KAMA above price, sell RSI>70 when KAMA below price.
# In trending markets (CHOP < 38.2): trend following - buy when price>KAMA and RSI 50-70, sell when price<KAMA and RSI 30-50.
# Uses volume confirmation to avoid false signals. Designed to work in both bull (trend following) and bear (mean reversion in chop) markets.
# Target: 75-200 total trades over 4 years (19-50/year) as per experiment guidelines.