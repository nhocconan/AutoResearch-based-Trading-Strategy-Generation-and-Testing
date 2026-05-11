#!/usr/bin/env python3
# 6h_ADX20_1dTrend_VolumeBreakout
# Hypothesis: On 6h timeframe, enter long when price breaks above 10-period high with ADX>20 (trending market),
# 1d trend is up (close > EMA50), and volume > 1.5x 20-period average. Enter short when price breaks below
# 10-period low with ADX>20, 1d trend down, and volume spike. Exit when price crosses back to 10-period
# midpoint or ADX drops below 20 (range market). ADX filters whipsaws in ranging markets, volume confirms
# breakout strength, and 1d trend ensures alignment with higher timeframe bias. Works in bull markets by
# catching strong uptrends and in bear by catching strong downtrends while avoiding chop.

name = "6h_ADX20_1dTrend_VolumeBreakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- ADX(14) calculation ---
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # First value is NaN
        
        # Directional Movement
        plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        plus_dm = np.concatenate([[np.nan], plus_dm])
        minus_dm = np.concatenate([[np.nan], minus_dm])
        
        # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
        def wilders_smooth(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            # First value is simple average
            result[period-1] = np.nanmean(data[1:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]):
                    result[i] = (result[i-1] * (period-1) + data[i]) / period
                else:
                    result[i] = np.nan
            return result
        
        tr_smooth = wilders_smooth(tr, period)
        plus_dm_smooth = wilders_smooth(plus_dm, period)
        minus_dm_smooth = wilders_smooth(minus_dm, period)
        
        # Directional Indicators
        plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
        minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
        
        # DX and ADX
        dx = np.where((plus_di + minus_di) != 0, 
                      100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = wilders_smooth(dx, period)
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # --- 10-period high/low for breakout ---
    high_10 = np.full(n, np.nan)
    low_10 = np.full(n, np.nan)
    for i in range(10, n):
        high_10[i] = np.max(high[i-10:i])
        low_10[i] = np.min(low[i-10:i])
    
    # --- 1d EMA50 trend ---
    close_1d = df_1d['close'].values
    ema_1d = np.full(len(close_1d), np.nan)
    for i in range(50, len(close_1d)):
        if i == 50:
            ema_1d[i] = np.mean(close_1d[0:50])
        else:
            ema_1d[i] = (close_1d[i] * 2 / (50 + 1)) + (ema_1d[i-1] * (49 / (50 + 1)))
    
    # Align 1d EMA to 6h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # --- Volume confirmation ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for ADX(14), 10-period HL, EMA50, and Vol MA(20)
    start_idx = max(14 + 14, 10, 50, 20)  # ADX needs ~28 bars for smoothing
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(adx[i]) or
            np.isnan(high_10[i]) or
            np.isnan(low_10[i]) or
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > high_10[i]
        breakout_down = close[i] < low_10[i]
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.5
        
        # Trend filter: 1d close vs EMA50
        trend_up = close_1d[-1] > ema_1d[-1] if len(close_1d) > 0 else False  # Current 1d bar
        trend_down = close_1d[-1] < ema_1d[-1] if len(close_1d) > 0 else False
        
        # Get current 1d trend from aligned value (more precise)
        # We need the 1d trend at the time of the 6h bar
        # Since we aligned the EMA, we can check if current close > EMA
        # But we need the 1d close that corresponds to this 6h bar
        # Simpler: use the aligned EMA and compare to current close
        # Actually, we want: is the 1d trend up/down?
        # We'll use: if the most recent completed 1d close > its EMA
        # To avoid look-ahead, we use the previous 1d bar's close vs EMA
        # But since we aligned, we can use: close[i] > ema_1d_aligned[i] for current bias
        # However, this uses current 6h close vs 1d EMA - not pure 1d trend
        # Better: get the 1d EMA value and compare to the 1d close that was available
        # We'll approximate: if the 6h close is above the aligned 1d EMA, bias is up
        # This is acceptable as the 1d EMA is slow
        trend_up = close[i] > ema_1d_aligned[i]
        trend_down = close[i] < ema_1d_aligned[i]
        
        if position == 0:
            if breakout_up and adx[i] > 20 and vol_spike and trend_up:
                # Long: upward breakout + trending ADX + volume spike + bullish 1d bias
                signals[i] = 0.25
                position = 1
            elif breakout_down and adx[i] > 20 and vol_spike and trend_down:
                # Short: downward breakout + trending ADX + volume spike + bearish 1d bias
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price falls to midpoint OR ADX weakens (range) OR trend fails
                midpoint = (high_10[i] + low_10[i]) / 2
                if close[i] < midpoint or adx[i] < 20 or not (close[i] > ema_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises to midpoint OR ADX weakens OR trend fails
                midpoint = (high_10[i] + low_10[i]) / 2
                if close[i] > midpoint or adx[i] < 20 or not (close[i] < ema_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals