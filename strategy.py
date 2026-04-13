#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot + volume spike + chop regime filter on 1h
    # Long: price > H3 AND volume > 1.5*volume_ma AND chop > 61.8 (range) → mean reversion short
    # Short: price < L3 AND volume > 1.5*volume_ma AND chop > 61.8 (range) → mean reversion long
    # Exit: price crosses H4/L4 or chop < 38.2 (trend) → follow trend
    # Using 4h for Camarilla pivots (structure), 1h for entry timing and volume/chop
    # Session filter: 08-20 UTC to avoid low-volume Asian session
    # Discrete position sizing (0.20) to minimize fee churn
    # Target: 15-37 trades/year (~60-150 over 4 years) to avoid fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla pivots (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (based on previous day's range)
    # Need daily OHLC from 4h data - resample to 1d using actual 4h bars
    # Since we cannot resample, we'll use 4h close as proxy for daily close
    # Better approach: use 4h bar's high/low/close for intraday Camarilla
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Camarilla pivot points for intraday trading
    # Based on previous 4h bar's range (more responsive than daily)
    # H4, H3, L3, L4 levels
    # Formula: 
    # Pivot = (high + low + close) / 3
    # Range = high - low
    # H4 = close + range * 1.1/2
    # H3 = close + range * 1.1/4
    # L3 = close - range * 1.1/4
    # L4 = close - range * 1.1/2
    
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    range_4h = high_4h - low_4h
    h4_4h = close_4h + range_4h * 1.1 / 2.0
    h3_4h = close_4h + range_4h * 1.1 / 4.0
    l3_4h = close_4h - range_4h * 1.1 / 4.0
    l4_4h = close_4h - range_4h * 1.1 / 2.0
    
    # Align 4h Camarilla levels to 1h (wait for completed 4h bar)
    h4_4h_aligned = align_htf_to_ltf(prices, df_4h, h4_4h)
    h3_4h_aligned = align_htf_to_ltf(prices, df_4h, h3_4h)
    l3_4h_aligned = align_htf_to_ltf(prices, df_4h, l3_4h)
    l4_4h_aligned = align_htf_to_ltf(prices, df_4h, l4_4h)
    
    # 1h indicators: volume MA and chopiness index
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Chopiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR(1)) / (n * ATR(14))) / log10(n)
    # where ATR(1) = True Range, ATR(14) = smoothed TR
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    atr_1 = tr  # True Range
    # Wilder's smoothing for ATR(14)
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_14 = wilders_smoothing(tr, 14)
    sum_atr_1 = np.nancumsum(atr_1)  # cumulative sum of TR
    n = 14
    chop = 100 * np.log10(sum_atr_1 / (n * atr_14)) / np.log10(n)
    chop = np.where(chop > 100, 100, chop)  # cap at 100
    chop = np.where(chop < 0, 0, chop)      # floor at 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # warmup for indicators
        # Skip if data not ready or outside session
        if (np.isnan(h3_4h_aligned[i]) or np.isnan(l3_4h_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(chop[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in ranging markets (chop > 61.8)
        ranging_market = chop[i] > 61.8
        trending_market = chop[i] < 38.2  # exit condition
        
        # Volume filter: volume spike
        volume_spike = volume[i] > 1.5 * volume_ma[i]
        
        # Mean reversion signals at Camarilla H3/L3 levels
        long_signal = (close[i] < l3_4h_aligned[i]) and ranging_market and volume_spike
        short_signal = (close[i] > h3_4h_aligned[i]) and ranging_market and volume_spike
        
        # Exit conditions: trend break or mean reversion completion
        long_exit = (close[i] > l4_4h_aligned[i]) or trending_market
        short_exit = (close[i] < h4_4h_aligned[i]) or trending_market
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_camarilla_volume_chop_v1"
timeframe = "1h"
leverage = 1.0