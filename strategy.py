#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_volatility_filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate ATR(14) on daily for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR(14) on weekly for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr_1w = np.maximum(
        high_1w[1:] - low_1w[1:],
        np.maximum(
            np.abs(high_1w[1:] - close_1w[:-1]),
            np.abs(low_1w[1:] - close_1w[:-1])
        )
    )
    tr_1w = np.concatenate([[np.nan], tr_1w])
    atr_14_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels on daily
    # Based on previous day's range
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # First day will have NaN from roll
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    range_1d = prev_high - prev_low
    camarilla_H4 = prev_close + 1.1 * range_1d / 2
    camarilla_L4 = prev_close - 1.1 * range_1d / 2
    camarilla_H3 = prev_close + 1.1 * range_1d / 4
    camarilla_L3 = prev_close - 1.1 * range_1d / 4
    
    # Align Camarilla levels and ATR to daily timeframe
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Volatility filter: current ATR > average ATR (avoid low volatility chop)
    atr_ma_1d = pd.Series(atr_14_1d_aligned).rolling(window=20, min_periods=20).mean().values
    volatility_ok = atr_14_1d_aligned > atr_ma_1d
    
    # Regime filter: weekly ATR > its 20-period average (avoid ranging markets on higher timeframe)
    atr_ma_1w = pd.Series(atr_14_1w_aligned).rolling(window=20, min_periods=20).mean().values
    regime_ok = atr_14_1w_aligned > atr_ma_1w
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # warmup for indicators
        # Skip if not ready
        if (np.isnan(camarilla_H4_aligned[i]) or np.isnan(camarilla_L4_aligned[i]) or
            np.isnan(camarilla_H3_aligned[i]) or np.isnan(camarilla_L3_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(atr_14_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions with filters
        breakout_above_H4 = close[i] > camarilla_H4_aligned[i]
        breakout_below_L4 = close[i] < camarilla_L4_aligned[i]
        
        # Mean reversion at H3/L3 levels
        mean_revert_H3 = close[i] < camarilla_H3_aligned[i] and close[i] > camarilla_L3_aligned[i]
        
        # Filters
        vol_filter = volatility_ok[i]
        regime_filter = regime_ok[i]
        
        # Entry signals: breakout with filters
        long_signal = breakout_above_H4 and vol_filter and regime_filter
        short_signal = breakout_below_L4 and vol_filter and regime_filter
        
        # Exit signals: mean reversion in middle range
        exit_long = mean_revert_H3 and position == 1
        exit_short = mean_revert_H3 and position == -1
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals