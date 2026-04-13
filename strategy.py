#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d daily KAMA trend with weekly ADX trend filter and volume confirmation.
# Uses KAMA (Kaufman Adaptive Moving Average) to capture trend direction while adapting to volatility.
# Weekly ADX > 25 ensures we only trade in strong trends, avoiding choppy markets.
# Volume confirmation ensures breakouts/instability have conviction.
# Target: 30-100 total trades over 4 years (7-25/year) to stay within profitable range.
# Works in both bull and bear markets by only trading when trend is strong (ADX filter).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for multi-timeframe analysis
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate KAMA on weekly close
    close_1w = df_1w['close'].values
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1w, n=10))
    volatility = np.sum(np.abs(np.diff(close_1w)), axis=1)
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Initialize KAMA
    kama = np.full_like(close_1w, np.nan, dtype=float)
    kama[9] = close_1w[9]  # Start after first 10 periods
    for i in range(10, len(close_1w)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # Calculate ADX on weekly
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    up_move = high_1w[1:] - high_1w[:-1]
    down_move = low_1w[:-1] - low_1w[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed values
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[:period]) / period
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilders_smooth(tr, 14)
    plus_di = 100 * wilders_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilders_smooth(minus_dm, 14) / atr
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = wilders_smooth(dx, 14)
    
    # Calculate weekly volume and its 20-period average
    volume_1w = df_1w['volume'].values
    volume_ma_20_1w = np.full_like(volume_1w, np.nan, dtype=float)
    for i in range(19, len(volume_1w)):
        volume_ma_20_1w[i] = np.mean(volume_1w[i-19:i+1])
    
    # Align all data to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    volume_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_20_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_ma_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current daily volume > 1.5x weekly volume MA (adjusted for daily)
        # Approximate: 5 trading days per week, so weekly MA/5 = approximate daily period MA
        volume_daily_approx_ma = volume_ma_20_1w_aligned[i] / 5
        volume_condition = volume[i] > (volume_daily_approx_ma * 1.5)
        
        # ADX condition: strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Entry conditions: KAMA trend with volume and ADX filter
        # Long when price above KAMA with volume and strong trend
        # Short when price below KAMA with volume and strong trend
        if position == 0:
            if close[i] > kama_aligned[i] and volume_condition and strong_trend:
                position = 1
                signals[i] = position_size
            elif close[i] < kama_aligned[i] and volume_condition and strong_trend:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price crosses below KAMA or trend weakens
            if close[i] < kama_aligned[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price crosses above KAMA or trend weakens
            if close[i] > kama_aligned[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_KAMA_ADX_Volume_Filter_v1"
timeframe = "1d"
leverage = 1.0