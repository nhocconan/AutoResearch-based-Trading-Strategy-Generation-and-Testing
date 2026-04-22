#!/usr/bin/env python3

"""
Hypothesis: Daily Donchian channel breakout with weekly trend filter and volume confirmation.
Only trade long when price breaks above Donchian upper band during low volatility (ATR contraction)
and weekly trend is up; short when price breaks below Donchian lower band during volatility contraction
and weekly trend is down. Uses ATR contraction to detect breakout readiness, avoiding false breakouts.
Designed for low trade frequency (7-25 trades/year) by requiring multiple confirmations: volatility contraction,
price breakout, and trend alignment. Works in both bull and bear markets by following the weekly trend.
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
    
    # Donchian Channel (20-day) on daily
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # ATR for volatility measurement
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR contraction: current ATR < 0.8 * 20-period ATR average (volatility compression)
    atr_ma_20 = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    vol_contract = atr < 0.8 * atr_ma_20
    
    # Load weekly data for trend filter - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Weekly EMA34 for trend direction
    weekly_close = df_weekly['close'].values
    ema34_weekly = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema34_weekly_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(vol_contract[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility contraction condition
        vol_cond = vol_contract[i]
        
        # Volume confirmation
        vol_spike = volume[i] > 1.3 * vol_ma_20[i]
        
        if position == 0:
            # Long: vol contraction + price breaks above upper band + weekly uptrend + volume spike
            if vol_cond and close[i] > donchian_upper[i] and ema34_weekly_aligned[i] > ema34_weekly_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: vol contraction + price breaks below lower band + weekly downtrend + volume spike
            elif vol_cond and close[i] < donchian_lower[i] and ema34_weekly_aligned[i] < ema34_weekly_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: volatility expansion (end of contraction) or price returns to middle band
            exit_signal = False
            
            if position == 1:
                # Exit long: volatility expansion or price closes below middle band
                if not vol_cond or close[i] < donchian_middle[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: volatility expansion or price closes above middle band
                if not vol_cond or close[i] > donchian_middle[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "Daily_Donchian_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0