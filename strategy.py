#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_RegimeFilter_v3
Hypothesis: Refine the Camarilla R1/S1 breakout with stricter volume confirmation (3.0x average) and add ATR-based volatility filter to avoid low-momentum whipsaws. Use 1d trend (close > EMA34) for directional bias. Target 20-35 trades/year by tightening entry conditions. Works in bull (breakouts with trend) and bear (fades false breakouts via regime filter).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need at least 34 periods for EMA34
        return np.zeros(n)
    
    # Calculate 1d OHLC for Camarilla pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (based on previous 1d bar's range)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_r1 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 12
    camarilla_s1 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 12
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align Camarilla levels and EMA to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 3.0x 20-period average (stricter)
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 3.0
    
    # ATR-based volatility filter: avoid low-volatility whipsaws
    def calculate_atr(high, low, close, window=14):
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        atr = pd.Series(tr).ewm(alpha=1/window, adjust=False, min_periods=window).mean().values
        return atr
    
    atr = calculate_atr(high, low, close, 14)
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    volatility_filter = atr > (0.5 * atr_ma)  # Only trade when volatility is above half its MA
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Camarilla (1d EMA34) + volume MA + ATR warmup
    start_idx = max(34, 20, 14)  # 34 for EMA34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1d trend alignment
        trend_1d_uptrend = close[i] > ema_34_1d_aligned[i]
        trend_1d_downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + 1d uptrend + volume spike + volatility filter
            long_signal = (close[i] > camarilla_r1_aligned[i]) and trend_1d_uptrend and volume_spike[i] and volatility_filter[i]
            
            # Short: price breaks below S1 + 1d downtrend + volume spike + volatility filter
            short_signal = (close[i] < camarilla_s1_aligned[i]) and trend_1d_downtrend and volume_spike[i] and volatility_filter[i]
            
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
            # Exit: price touches S1 OR 1d trend turns down
            if (close[i] < camarilla_s1_aligned[i] or not trend_1d_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price touches R1 OR 1d trend turns up
            if (close[i] > camarilla_r1_aligned[i] or not trend_1d_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_RegimeFilter_v3"
timeframe = "4h"
leverage = 1.0