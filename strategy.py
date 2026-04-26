#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1wTrend_1dRegime_v1
Hypothesis: On 12h timeframe, trade Camarilla R1/S1 breakouts with 1w EMA50 trend filter and 1d chop regime (mean reversion when choppy, trend following when trending). Volume confirmation reduces false breakouts. Designed for low trade frequency (12-37/year) to minimize fee drag while capturing swings in both bull and bear markets via multi-timeframe confluence.
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
    
    # Get 1w data for HTF trend (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1d data for Camarilla pivot levels (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    camarilla_r1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    camarilla_s1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # Get 1d data for chop regime filter (Choppiness Index)
    # CHOP(14) = 100 * log10(sum(ATR(14)) / log10(range(14))) / log10(14)
    # Simplified: use ATR ratio for regime detection
    tr1 = pd.Series(df_1d['high']).rolling(window=2).max() - pd.Series(df_1d['low']).rolling(window=2).min()
    tr2 = abs(pd.Series(df_1d['high']).rolling(window=2).max() - pd.Series(df_1d['close']).shift(1))
    tr3 = abs(pd.Series(df_1d['low']).rolling(window=2).min() - pd.Series(df_1d['close']).shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr_1d.rolling(window=14, min_periods=14).mean().values
    high_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    low_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr_14.sum() / (high_14 - low_14)) / np.log10(14) if (high_14 - low_14) > 0 else 50
    # For simplicity, use ATR normalization as proxy: chop > 0.6 = choppy, < 0.4 = trending
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    chop_ratio = atr_14 / atr_ma_50
    chop_ratio = np.where(np.isnan(chop_ratio), 0.5, chop_ratio)
    
    # Volume spike: current 12h volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Align HTF indicators to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1, additional_delay_bars=1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1, additional_delay_bars=1)
    chop_ratio_aligned = align_htf_to_ltf(prices, df_1d, chop_ratio, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of EMA(50) 1w (50), Camarilla (need 2nd 1d bar), volume MA (20), chop ratio (50)
    start_idx = max(50, 2, 20, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(chop_ratio_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_1w_val = ema_50_1w_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol_spike = volume_spike[i]
        camarilla_r1_val = camarilla_r1_aligned[i]
        camarilla_s1_val = camarilla_s1_aligned[i]
        chop_val = chop_ratio_aligned[i]
        
        # Regime filter: chop > 0.6 = choppy (mean revert), chop < 0.4 = trending (trend follow)
        is_choppy = chop_val > 0.6
        is_trending = chop_val < 0.4
        
        if position == 0:
            # Long conditions
            long_breakout = high_val > camarilla_r1_val
            long_trend = close_val > ema_50_1w_val  # uptrend filter
            long_mean_revert = low_val < camarilla_s1_val and close_val > camarilla_s1_val  # bounce from S1
            
            # Short conditions
            short_breakout = low_val < camarilla_s1_val
            short_trend = close_val < ema_50_1w_val  # downtrend filter
            short_mean_revert = high_val > camarilla_r1_val and close_val < camarilla_r1_val  # rejection from R1
            
            # In choppy regime: mean reversion at S1/R1
            # In trending regime: breakout in direction of trend
            if is_choppy:
                long_signal = long_mean_revert and vol_spike
                short_signal = short_mean_revert and vol_spike
            else:  # trending or neutral
                long_signal = long_breakout and long_trend and vol_spike
                short_signal = short_breakout and short_trend and vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: trend reversal or mean reversion signal
            if is_choppy and close_val > camarilla_r1_val:  # reached R1 in choppy market
                signals[i] = 0.0
                position = 0
            elif not is_choppy and close_val < ema_50_1w_val:  # trend reversal
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend reversal or mean reversion signal
            if is_choppy and close_val < camarilla_s1_val:  # reached S1 in choppy market
                signals[i] = 0.0
                position = 0
            elif not is_choppy and close_val > ema_50_1w_val:  # trend reversal
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1wTrend_1dRegime_v1"
timeframe = "12h"
leverage = 1.0