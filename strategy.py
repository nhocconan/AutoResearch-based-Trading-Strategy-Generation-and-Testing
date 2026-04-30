#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h ADX20 trend filter and volume spike confirmation
# Camarilla R1/S1 levels provide intraday support/resistance for precise 1h entries
# 4h ADX > 20 ensures alignment with intermediate trend to avoid counter-trend whipsaws
# Volume spike (1.8x 50-period average) confirms participation on 1h timeframe
# Discrete sizing 0.20 controls risk and minimizes fee churn. Target: 80-120 total trades over 4 years (20-30/year).
# Works in bull markets via breakouts above R1 and bear markets via breakdowns below S1 with trend filter.
# Session filter (08-20 UTC) reduces noise trades during low-volume periods.

name = "1h_Camarilla_R1S1_4hADX20_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop (MTF Rule #1)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h ADX for trend filter
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Calculate +DM and -DM
    up_move = high_4h - np.roll(high_4h, 1)
    down_move = np.roll(low_4h, 1) - low_4h
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    atr_4h = wilder_smooth(tr, period)
    plus_di_4h = 100 * wilder_smooth(plus_dm, period) / atr_4h
    minus_di_4h = 100 * wilder_smooth(minus_dm, period) / atr_4h
    dx_4h = 100 * np.abs(plus_di_4h - minus_di_4h) / (plus_di_4h + minus_di_4h)
    adx_4h = wilder_smooth(dx_4h, period)
    
    # Align 4h ADX to 1h timeframe
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Calculate 4h Camarilla pivot levels (R1, S1)
    camarilla_r1 = close_4h + ((high_4h - low_4h) * 1.125 / 4)
    camarilla_s1 = close_4h - ((high_4h - low_4h) * 1.125 / 4)
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Volume confirmation: volume > 1.8x 50-period average (50*1h = ~2 days)
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (1.8 * vol_ma_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 50)  # warmup for ADX and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(adx_4h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma_50[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_adx = adx_4h_aligned[i]
        curr_r1 = camarilla_r1_aligned[i]
        curr_s1 = camarilla_s1_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and intermediate trend (ADX > 20)
            if curr_volume_spike and curr_adx > 20:
                # Bullish entry: break above R1
                if curr_close > curr_r1:
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                # Bearish entry: break below S1
                elif curr_close < curr_s1:
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below R1 (breakout fails) OR trend weakens (ADX < 15)
            if curr_close < curr_r1 or curr_adx < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit when price rises above S1 (breakdown fails) OR trend weakens (ADX < 15)
            if curr_close > curr_s1 or curr_adx < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals