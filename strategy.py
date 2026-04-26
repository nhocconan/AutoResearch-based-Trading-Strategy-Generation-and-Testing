#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1wTrend_ADXFilter_v1
Hypothesis: 4h Camarilla R1/S1 breakout with 1w EMA34 trend filter and ADX>20 for trend strength.
- Uses 4h timeframe targeting 75-200 total trades over 4 years (19-50/year)
- Long when price breaks above R1 with 1w uptrend (EMA34) and ADX>20
- Short when price breaks below S1 with 1w downtrend (EMA34) and ADX>20
- Camarilla levels derived from previous 1d OHLC for structure-aware entries
- 1w trend filter ensures we trade with the dominant weekly trend
- ADX filter ensures we only trade when trend strength is sufficient
- Designed for low trade frequency with proven edge on BTC/ETH from historical data
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (wait for completed 1d bar)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Load 1w data ONCE before loop for EMA34 and ADX
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate ADX on 1w for trend strength filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_1w[0] - low_1w[0]  # First TR
    
    # Directional Movement
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Wilder's smoothing
    def wilders_smoothing(values, period):
        result = np.zeros_like(values)
        if len(values) < period:
            return result
        result[period-1] = np.nansum(values[:period])  # First value is simple average
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    period = 14
    atr_1w = wilders_smoothing(tr, period)
    plus_di_1w = 100 * wilders_smoothing(plus_dm, period) / (atr_1w + 1e-10)
    minus_di_1w = 100 * wilders_smoothing(minus_dm, period) / (atr_1w + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w + 1e-10)
    adx_1w = wilders_smoothing(dx, period)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34+14 for EMA34 and ADX)
    start_idx = max(34 + 14, 0)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(adx_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla breakout conditions with trend filters
        price_above_R1 = close[i] > R1_aligned[i]
        price_below_S1 = close[i] < S1_aligned[i]
        
        # 1w trend filter
        trend_up = close[i] > ema34_1w_aligned[i]
        trend_down = close[i] < ema34_1w_aligned[i]
        
        # ADX filter: only trade when trend is sufficient (ADX > 20)
        sufficient_trend = adx_1w_aligned[i] > 20.0
        
        if position == 0:
            # Long: price breaks above R1 AND 1w uptrend AND sufficient trend
            if price_above_R1 and trend_up and sufficient_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND 1w downtrend AND sufficient trend
            elif price_below_S1 and trend_down and sufficient_trend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below S1 OR 1w trend turns down OR trend weakens
            if price_below_S1 or not trend_up or not sufficient_trend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above R1 OR 1w trend turns up OR trend weakens
            if price_above_R1 or not trend_down or not sufficient_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1wTrend_ADXFilter_v1"
timeframe = "4h"
leverage = 1.0