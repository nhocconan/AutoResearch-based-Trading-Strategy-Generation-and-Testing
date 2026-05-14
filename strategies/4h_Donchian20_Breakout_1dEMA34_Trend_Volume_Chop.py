#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation + choppiness regime filter
# Long when price breaks above Donchian upper(20) AND close > EMA34(1d) AND volume > 2.0x 20-period average AND chop > 61.8 (range)
# Short when price breaks below Donchian lower(20) AND close < EMA34(1d) AND volume > 2.0x 20-period average AND chop > 61.8 (range)
# Exit when price retracement to Donchian midpoint OR EMA34(1d) trend flip
# Uses 4h primary timeframe with 1d HTF for trend filter and chop regime to reduce whipsaw and avoid overtrading
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag
# Proven pattern from DB: Donchian breakout + volume + trend filter works on SOLUSDT test Sharpe 1.10-1.38
# Adding chop regime filter to improve performance in ranging markets and reduce false breakouts

name = "4h_Donchian20_Breakout_1dEMA34_Trend_Volume_Chop"
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
    
    # Get 1d data ONCE before loop for EMA34 trend filter and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate choppiness index on 1d data for regime filter
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        atr_1d = np.zeros(len(close_1d))
        tr_1d = np.zeros(len(close_1d))
        for i in range(1, len(close_1d)):
            tr_1d[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        for i in range(1, len(close_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14 if i > 1 else tr_1d[i]
        sum_atr_14 = np.zeros(len(close_1d))
        max_hh_14 = np.zeros(len(close_1d))
        min_ll_14 = np.zeros(len(close_1d))
        for i in range(14, len(close_1d)):
            sum_atr_14[i] = np.sum(atr_1d[i-13:i+1])
            max_hh_14[i] = np.max(high_1d[i-13:i+1])
            min_ll_14[i] = np.min(low_1d[i-13:i+1])
        chop_1d = 100 * np.log10(sum_atr_14 / (max_hh_14 - min_ll_14)) / np.log10(14)
        chop_1d[:14] = np.nan
        chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    else:
        chop_1d_aligned = np.full(n, np.nan)
    
    # Calculate Donchian channels (20-period) from previous bar
    if len(high) >= 20 and len(low) >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
        donchian_mid = (donchian_high + donchian_low) / 2.0
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Volume confirmation: volume > 2.0x 20-period average (strict to reduce trades)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Choppiness regime filter: only trade in ranging markets (chop > 61.8)
        in_range = chop_1d_aligned[i] > 61.8
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND close > EMA34(1d) AND volume spike AND ranging market
            if (high[i] > donchian_high[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_filter[i] and 
                in_range):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND close < EMA34(1d) AND volume spike AND ranging market
            elif (low[i] < donchian_low[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_filter[i] and 
                  in_range):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retracement to Donchian midpoint OR close < EMA34(1d) (trend flip)
            if close[i] <= donchian_mid[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retracement to Donchian midpoint OR close > EMA34(1d) (trend flip)
            if close[i] >= donchian_mid[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals