#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_ATRRegime_v2
Hypothesis: Camarilla R1/S1 breakouts filtered by 1d EMA34 trend and ATR-based regime (low volatility = mean reversion, high volatility = trend follow) to avoid chop whipsaws. Uses discrete sizing (0.0, ±0.25) for optimal 4h trade frequency (target: 50-150/4 years). Works in bull/bear via trend filter + regime adaptation.
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
    
    # Get 1d data for HTF trend filter and ATR regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # need enough for EMA34 and ATR
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d ATR(14) for volatility regime
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d_series = pd.Series(df_1d['close'].values)
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d_series.shift(1))
    tr3 = abs(low_1d - close_1d_series.shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 1d ATR percentile rank (20-day lookback) for regime
    atr_percentile = pd.Series(atr_14_1d).rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # Calculate 1d OHLC for Camarilla pivot levels (previous day)
    o_1d = df_1d['open'].values
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = c_1d + (h_1d - l_1d) * 1.1 / 12
    camarilla_s1 = c_1d - (h_1d - l_1d) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup + ATR percentile
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(atr_percentile_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # ATR regime: <30 = low vol (mean reversion), >70 = high vol (trend follow)
        vol_regime = atr_percentile_aligned[i]
        is_low_vol = vol_regime < 30
        is_high_vol = vol_regime > 70
        
        if position == 0:
            # Long conditions: price > 1d EMA34 + breaks above R1
            # In low vol: mean reversion at S1 (long when price < S1)
            # In high vol: trend follow (long when price > R1)
            if is_low_vol:
                long_signal = (close[i] < camarilla_s1_aligned[i])  # mean reversion long
            else:  # high vol or neutral
                long_signal = (close[i] > ema_34_1d_aligned[i] and 
                              close[i] > camarilla_r1_aligned[i])  # trend long
            
            # Short conditions: price < 1d EMA34 + breaks below S1
            # In low vol: mean reversion at R1 (short when price > R1)
            # In high vol: trend follow (short when price < S1)
            if is_low_vol:
                short_signal = (close[i] > camarilla_r1_aligned[i])  # mean reversion short
            else:  # high vol or neutral
                short_signal = (close[i] < ema_34_1d_aligned[i] and 
                               close[i] < camarilla_s1_aligned[i])  # trend short
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit conditions: adaptive based on regime
            if is_low_vol:
                # Mean reversion: exit at midpoint or opposite level
                exit_signal = (close[i] > (camarilla_r1_aligned[i] + camarilla_s1_aligned[i]) / 2)
            else:
                # Trend follow: exit on trend break or opposite Camarilla touch
                exit_signal = (close[i] < ema_34_1d_aligned[i] or 
                              close[i] < camarilla_s1_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit conditions: adaptive based on regime
            if is_low_vol:
                # Mean reversion: exit at midpoint or opposite level
                exit_signal = (close[i] < (camarilla_r1_aligned[i] + camarilla_s1_aligned[i]) / 2)
            else:
                # Trend follow: exit on trend break or opposite Camarilla touch
                exit_signal = (close[i] > ema_34_1d_aligned[i] or 
                              close[i] > camarilla_r1_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_ATRRegime_v2"
timeframe = "4h"
leverage = 1.0