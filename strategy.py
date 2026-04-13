#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla H3/L3 breakout with 1d ATR volatility filter and volume confirmation
    # Long: price breaks above H3 AND ATR(14) > 1.2x 50-period median ATR (high volatility) AND volume > 1.5x avg
    # Short: price breaks below L3 AND ATR(14) > 1.2x 50-period median ATR AND volume > 1.5x avg
    # Exit: price touches H4 (for longs) or L4 (for shorts) OR price retests the breakout level (H3/L3)
    # Using 12h timeframe for optimal trade frequency (target 12-37/year), ATR filter to avoid low-volatility false breakouts,
    # and volume confirmation to ensure participation. Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Wilder's smoothing for ATR
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: smoothed = (prev * (period-1) + current) / period
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr14 = wilders_smoothing(tr, 14)
    
    # Calculate 50-period median ATR for volatility regime filter
    atr_median = np.full_like(atr14, np.nan)
    for i in range(50, len(atr14)):
        atr_median[i] = np.nanmedian(atr14[i-50:i])
    
    # Volatility filter: current ATR > 1.2x median ATR (high volatility regime)
    volatility_filter = atr14 > (1.2 * atr_median)
    
    # Align daily ATR and volatility filter to 12h
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    volatility_filter_aligned = align_htf_to_ltf(prices, df_1d, volatility_filter)
    
    # Get previous day's OHLC for Camarilla calculation
    df_1d_ohlc = get_htf_data(prices, '1d')
    if len(df_1d_ohlc) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation (shifted by 1 to avoid look-ahead)
    prev_close = df_1d_ohlc['close'].shift(1).values
    prev_high = df_1d_ohlc['high'].shift(1).values
    prev_low = df_1d_ohlc['low'].shift(1).values
    
    # Camarilla levels: H3, L3, H4, L4
    camarilla_h3 = prev_close + 1.25 * (prev_high - prev_low)
    camarilla_l3 = prev_close - 1.25 * (prev_high - prev_low)
    camarilla_h4 = prev_close + 1.5 * (prev_high - prev_low)
    camarilla_l4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Align Camarilla levels to 12h
    h3_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_l4)
    
    # Get 12h volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(atr_14_aligned[i]) or np.isnan(volatility_filter_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade in high volatility regimes
        high_volatility = volatility_filter_aligned[i]
        
        # Camarilla breakout conditions
        breakout_h3 = close[i] > h3_aligned[i]
        breakout_l3 = close[i] < l3_aligned[i]
        
        # Exit conditions: touch H4/L4 or price retests the breakout level (H3/L3)
        touch_h4 = close[i] >= h4_aligned[i]
        touch_l4 = close[i] <= l4_aligned[i]
        retest_h3 = close[i] < h3_aligned[i] and position == 1  # Long exit on H3 retest
        retest_l3 = close[i] > l3_aligned[i] and position == -1  # Short exit on L3 retest
        
        # Entry logic: Camarilla breakout + high volatility + volume confirmation
        long_entry = breakout_h3 and high_volatility and volume_spike[i]
        short_entry = breakout_l3 and high_volatility and volume_spike[i]
        
        # Exit logic: H4/L4 touch or retest of breakout level
        long_exit = touch_h4 or retest_l3
        short_exit = touch_l4 or retest_h3
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_camarilla_h3l3_breakout_atr_volume_v1"
timeframe = "12h"
leverage = 1.0