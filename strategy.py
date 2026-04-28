#!/usr/bin/env python3
"""
4h_KAMA_Trend_VolumeSpike_RSI4060
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise, providing a strong trend filter in both bull and bear markets. Combined with volume spike (2x 20-bar average) and RSI in 40-60 range (avoiding extremes), this strategy captures trending moves while avoiding chop. Uses 1d trend filter for higher timeframe confirmation. Targets 20-30 trades/year via strict conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate KAMA (20, 2, 30) on close
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder
    # Proper volatility calculation: sum of absolute changes over ER period
    er_period = 10
    change_arr = np.abs(np.diff(close, prepend=close[0]))
    volatility_arr = np.convolve(change_arr, np.ones(er_period), 'same')  # sum of abs changes over er_period
    volatility_arr[:er_period-1] = np.nan  # not enough data
    er = np.where(volatility_arr != 0, change_arr / volatility_arr, 0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2  # fast SC=2/(2+1)=0.6645, slow SC=2/(30+1)=0.0645
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(sc[i]) or np.isnan(kama[i-1]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: >2x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA34 and KAMA to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(kama[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # KAMA direction: price above/below KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI in neutral zone (40-60) to avoid extremes
        rsi_neutral = (rsi[i] >= 40) & (rsi[i] <= 60)
        
        # Volume confirmation (>2x average)
        vol_confirm = volume[i] > (2.0 * vol_ma_20[i])
        
        # Entry conditions
        long_entry = uptrend & price_above_kama & rsi_neutral & vol_confirm
        short_entry = downtrend & price_below_kama & rsi_neutral & vol_confirm
        
        # Exit conditions: opposite KAMA cross
        long_exit = price_below_kama
        short_exit = price_above_kama
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_KAMA_Trend_VolumeSpike_RSI4060"
timeframe = "4h"
leverage = 1.0