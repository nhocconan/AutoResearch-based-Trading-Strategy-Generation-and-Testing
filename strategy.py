# 2025-07-03 02:30:00
#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Regime
Hypothesis: Combines Camarilla R1/S1 breakouts with 1d EMA trend filter, volume confirmation (>1.8x 20-bar avg), and Choppiness Index regime filter (CHOP>61.8 for mean reversion, CHOP<38.2 for trend following). Uses tighter volume threshold and regime filter to reduce overtrading vs prior variants. Designed for 4h to work in both bull (breakouts) and bear (mean reversion in ranges) markets.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Regime"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar (based on previous day's range)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    valid_idx = ~np.isnan(prev_high) & ~np.isnan(prev_low) & ~np.isnan(prev_close)
    camarilla_r1 = np.full_like(prev_close, np.nan)
    camarilla_s1 = np.full_like(prev_close, np.nan)
    
    camarilla_r1[valid_idx] = prev_close[valid_idx] + 1.1 * (prev_high[valid_idx] - prev_low[valid_idx]) / 12
    camarilla_s1[valid_idx] = prev_close[valid_idx] - 1.1 * (prev_high[valid_idx] - prev_low[valid_idx]) / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Get 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.8x 20-period average (tighter than 2.0x)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.8 * vol_ma)
    
    # Choppiness Index regime filter (using 1d data)
    # CHOP = 100 * log10(sum(ATR over n) / (max(high,n) - min(low,n))) / log10(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_1d = np.maximum(np.abs(high_1d[1:] - low_1d[:-1]),
                        np.maximum(np.abs(high_1d[1:] - close_1d[:-1]),
                                   np.abs(low_1d[1:] - close_1d[:-1])))
    atr_1d = np.concatenate([[np.nan], atr_1d])  # align with index
    
    chop_window = 14
    sum_atr = pd.Series(atr_1d).rolling(window=chop_window, min_periods=chop_window).sum().values
    max_high = pd.Series(high_1d).rolling(window=chop_window, min_periods=chop_window).max().values
    min_low = pd.Series(low_1d).rolling(window=chop_window, min_periods=chop_window).min().values
    chop = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(chop_window)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Regime: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending
    ranging = chop_aligned > 61.8
    trending = chop_aligned < 38.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        if position == 0:
            # LONG conditions
            long_breakout = (not np.isnan(camarilla_r1_aligned[i]) and 
                            high[i] > camarilla_r1_aligned[i])
            long_trend = close[i] > ema_34_1d_aligned[i]
            long_volume = volume_confirmed[i]
            
            # SHORT conditions
            short_breakout = (not np.isnan(camarilla_s1_aligned[i]) and 
                             low[i] < camarilla_s1_aligned[i])
            short_trend = close[i] < ema_34_1d_aligned[i]
            short_volume = volume_confirmed[i]
            
            # In ranging markets: mean reversion at extremes
            if ranging[i]:
                if long_breakout and short_trend and long_volume:  # Oversold bounce in range
                    signals[i] = 0.25
                    position = 1
                elif short_breakout and long_trend and short_volume:  # Overbought rejection in range
                    signals[i] = -0.25
                    position = -1
            # In trending markets: breakout with trend
            elif trending[i]:
                if long_breakout and long_trend and long_volume:
                    signals[i] = 0.25
                    position = 1
                elif short_breakout and short_trend and short_volume:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # EXIT LONG: price breaks below S1 or trend fails
            if (not np.isnan(camarilla_s1_aligned[i]) and low[i] < camarilla_s1_aligned[i]) or \
               close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above R1 or trend fails
            if (not np.isnan(camarilla_r1_aligned[i]) and high[i] > camarilla_r1_aligned[i]) or \
               close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals