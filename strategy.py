#!/usr/bin/env python3
"""
6h_Camarilla_R1_S1_Breakout_12hTrend_ADXFilter_v1
Hypothesis: 6h Camarilla R1/S1 breakout with 12h EMA50 trend filter and ADX>25 for trend strength.
- Uses 6h timeframe targeting 50-150 total trades over 4 years (12-37/year)
- Long when price breaks above R1 with ADX>25 and 12h uptrend (price>EMA50)
- Short when price breaks below S1 with ADX>25 and 12h downtrend (price<EMA50)
- Camarilla levels derived from previous 6h OHLC for structure-aware entries
- ADX filter ensures we only trade in trending markets, avoiding whipsaws in ranges
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
    
    # Load 6h data ONCE before loop for Camarilla levels, EMA50, and ADX
    df_6h = get_htf_data(prices, '6h')
    
    # Calculate Camarilla levels from previous 6h bar
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_close = df_6h['close'].values
    prev_high = df_6h['high'].values
    prev_low = df_6h['low'].values
    
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 6h timeframe (wait for completed 6h bar)
    R1_aligned = align_htf_to_ltf(prices, df_6h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_6h, S1)
    
    # Calculate 6h EMA50 for trend filter
    ema50_6h = pd.Series(df_6h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_6h_aligned = align_htf_to_ltf(prices, df_6h, ema50_6h)
    
    # Calculate ADX on 6h for trend strength filter
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # True Range
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_6h[0] - low_6h[0]  # First TR
    
    # Directional Movement
    up_move = high_6h - np.roll(high_6h, 1)
    down_move = np.roll(low_6h, 1) - low_6h
    
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
    atr_6h = wilders_smoothing(tr, period)
    plus_di_6h = 100 * wilders_smoothing(plus_dm, period) / (atr_6h + 1e-10)
    minus_di_6h = 100 * wilders_smoothing(minus_dm, period) / (atr_6h + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_6h - minus_di_6h) / (plus_di_6h + minus_di_6h + 1e-10)
    adx_6h = wilders_smoothing(dx, period)
    adx_6h_aligned = align_htf_to_ltf(prices, df_6h, adx_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50+14 for EMA50 and ADX)
    start_idx = max(50 + 14, 0)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_6h_aligned[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(adx_6h_aligned[i])):
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
        
        # 6h trend filter
        trend_up = close[i] > ema50_6h_aligned[i]
        trend_down = close[i] < ema50_6h_aligned[i]
        
        # ADX filter: only trade when trend is strong (ADX > 25)
        strong_trend = adx_6h_aligned[i] > 25.0
        
        if position == 0:
            # Long: price breaks above R1 AND 12h uptrend AND strong trend
            if price_above_R1 and trend_up and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND 12h downtrend AND strong trend
            elif price_below_S1 and trend_down and strong_trend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below S1 OR 12h trend turns down OR trend weakens
            if price_below_S1 or not trend_up or not strong_trend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above R1 OR 12h trend turns up OR trend weakens
            if price_above_R1 or not trend_down or not strong_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R1_S1_Breakout_12hTrend_ADXFilter_v1"
timeframe = "6h"
leverage = 1.0