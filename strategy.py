#!/usr/bin/env python3
"""
4h_CRSI_Donchian_Breakout_Volume_Chop
Hypothesis: 4-hour Connors RSI (CRSI) with Donchian(20) breakout, volume confirmation (>1.5x 20-period average), and choppiness regime filter (CHOP > 61.8 for mean reversion, < 38.2 for trend following). Long when CRSI < 15 and price breaks above Donchian upper band in low-chop regime; short when CRSI > 85 and price breaks below Donchian lower band in low-chop regime. Exit via opposite Donchian band or ATR trailing stop (2.5*ATR from extreme). Designed for ~100-180 trades over 4 years (25-45/year) via tight CRSI extremes + Donchian breakout confluence.
"""

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
    
    # Get 1d data for choppiness filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # need sufficient data for chop calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d ATR(14) for choppiness
    atr_period = 14
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Calculate 1d choppiness index: CHOP = 100 * log10(sum(ATR14)/ (n * (HHV - LLV))) / log10(n)
    chop_period = 14
    sum_atr = pd.Series(atr_1d).rolling(window=chop_period, min_periods=chop_period).sum().values
    hh = pd.Series(high_1d).rolling(window=chop_period, min_periods=chop_period).max().values
    ll = pd.Series(low_1d).rolling(window=chop_period, min_periods=chop_period).min().values
    chop_raw = 100 * np.log10(sum_atr / (chop_period * (hh - ll))) / np.log10(chop_period)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    # Choppiness regime: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trending (trend follow)
    chop_range = chop_aligned > 61.8
    chop_trend = chop_aligned < 38.2
    
    # Calculate Connors RSI (CRSI) on 4h data
    # CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    rsi_period = 3
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # RSI Streak: consecutive up/down days
    streak = np.zeros(len(close))
    streak[0] = 0
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    # RSI of streak (capped at 2 for CRSI)
    streak_abs = np.minimum(np.abs(streak), 2)
    streak_rsi = np.zeros(len(close))
    for i in range(len(close)):
        if streak[i] > 0:
            streak_rsi[i] = min(100 * streak_abs[i] / 2, 100)  # up streak
        elif streak[i] < 0:
            streak_rsi[i] = 100 - min(100 * streak_abs[i] / 2, 100)  # down streak
        else:
            streak_rsi[i] = 50  # no streak
    
    # Percent Rank (100-period)
    lookback = 100
    percent_rank = np.zeros(len(close))
    for i in range(lookback, len(close)):
        window = close[i-lookback:i]
        percent_rank[i] = (np.sum(window < close[i]) / lookback) * 100
    
    # CRSI calculation
    crsi = (rsi_values + streak_rsi + percent_rank) / 3
    
    # Donchian(20) channels
    donch_period = 20
    upper_band = pd.Series(high).rolling(window=donch_period, min_periods=donch_period).max().values
    lower_band = pd.Series(low).rolling(window=donch_period, min_periods=donch_period).min().values
    
    # ATR for stoploss (14-period)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume regime: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0   # highest close since long entry
    short_extreme = 0.0  # lowest close since short entry
    
    # Start index: need warmup for calculations
    start_idx = max(100, donch_period, atr_period, 20, lookback)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(crsi[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long conditions: CRSI < 15 (oversold) + price breaks above upper band + volume + chop trend
            long_signal = (crsi[i] < 15) and (close[i] > upper_band[i]) and vol_regime[i] and chop_trend[i]
            # Short conditions: CRSI > 85 (overbought) + price breaks below lower band + volume + chop trend
            short_signal = (crsi[i] > 85) and (close[i] < lower_band[i]) and vol_regime[i] and chop_trend[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                long_extreme = close[i]
            elif short_signal:
                signals[i] = -0.25
                position = -1
                short_extreme = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Update extreme for trailing stop
            if close[i] > long_extreme:
                long_extreme = close[i]
            # Exit conditions: 
            # 1. ATR trailing stop (2.5*ATR from extreme)
            atr_stop = long_extreme - 2.5 * atr[i]
            # 2. Price breaks below lower Donchian band
            if close[i] <= atr_stop or close[i] < lower_band[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Update extreme for trailing stop
            if close[i] < short_extreme:
                short_extreme = close[i]
            # Exit conditions:
            # 1. ATR trailing stop (2.5*ATR from extreme)
            atr_stop = short_extreme + 2.5 * atr[i]
            # 2. Price breaks above upper Donchian band
            if close[i] >= atr_stop or close[i] > upper_band[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_CRSI_Donchian_Breakout_Volume_Chop"
timeframe = "4h"
leverage = 1.0