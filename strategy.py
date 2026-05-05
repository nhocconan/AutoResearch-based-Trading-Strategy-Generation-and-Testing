#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R Extreme with 1d EMA34 trend filter and volume spike confirmation
# Long when Williams %R < -80 (oversold) AND price > EMA34(1d) (uptrend) AND volume > 2.0x 20-period average
# Short when Williams %R > -20 (overbought) AND price < EMA34(1d) (downtrend) AND volume > 2.0x 20-period average
# Exit when Williams %R returns to neutral zone (-50) OR trend flips
# Uses discrete sizing (0.25) to limit fee drag. Target: 20-50 trades/year per symbol.
# Williams %R identifies exhaustion points that work in both bull (buy oversold in uptrend) and bear (sell overbought in downtrend) markets.
# 1d EMA34 provides higher timeframe trend filter to avoid counter-trend whipsaws, extreme volume spike confirms institutional participation.

name = "4h_WilliamsR_Extreme_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R on 4h data: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Williams %R = -100 * (HHV - Close) / (HHV - LLV) where HHV/HLV are over lookback period
    lookback = 14
    highest_high = pd.Series(close).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(close).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: volume > 2.0x 20-period average (extreme spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) AND price > EMA34(1d) AND volume spike
            if (williams_r[i] < -80 and 
                close[i] > ema_34_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (overbought) AND price < EMA34(1d) AND volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema_34_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to neutral zone (> -50) OR price < EMA34(1d) (trend flip)
            if (williams_r[i] > -50 or 
                close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to neutral zone (< -50) OR price > EMA34(1d) (trend flip)
            if (williams_r[i] < -50 or 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals