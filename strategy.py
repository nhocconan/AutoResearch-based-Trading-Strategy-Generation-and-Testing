#!/usr/bin/env python3
"""
1d_1w_Camilla_Trend_Filter
Hypothesis: Combines Camarilla pivot levels on daily with weekly trend filter to trade reversions in range-bound markets and trend continuations in trending markets.
In ranging markets (weekly ADX < 25), trades mean reversion at Camarilla support/resistance levels.
In trending markets (weekly ADX >= 25), trades breakouts of Camarilla expansion levels.
Uses volume confirmation to avoid false signals. Designed for low-frequency trading (10-25 trades/year) to minimize fee impact.
Works in both bull and bear markets by adapting to market regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for daily
    # Using previous day's OHLC
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]  # first day uses current day
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    # Camarilla calculations
    range_ = prev_high - prev_low
    camarilla_h5 = prev_close + 1.1 * range_ * 1.1 / 12  # Resistance 5
    camarilla_h4 = prev_close + 1.1 * range_ * 1.1 / 6   # Resistance 4
    camarilla_h3 = prev_close + 1.1 * range_ * 1.1 / 4   # Resistance 3
    camarilla_l3 = prev_close - 1.1 * range_ * 1.1 / 4   # Support 3
    camarilla_l4 = prev_close - 1.1 * range_ * 1.1 / 6   # Support 4
    camarilla_l5 = prev_close - 1.1 * range_ * 1.1 / 12  # Support 5
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly ADX (14)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            elif plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            else:
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], 
                       abs(high[i] - high[i-1]), 
                       abs(low[i] - low[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        
        # Initial values
        atr[period-1] = np.mean(tr[1:period])
        plus_di[period-1] = np.mean(plus_dm[1:period]) / atr[period-1] * 100 if atr[period-1] != 0 else 0
        minus_di[period-1] = np.mean(minus_dm[1:period]) / atr[period-1] * 100 if atr[period-1] != 0 else 0
        
        # Wilder smoothing
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_di[i] = (plus_di[i-1] * (period-1) + plus_dm[i]) / atr[i] * 100 if atr[i] != 0 else 0
            minus_di[i] = (minus_di[i-1] * (period-1) + minus_dm[i]) / atr[i] * 100 if atr[i] != 0 else 0
        
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = np.zeros_like(high)
        adx[2*period-2] = np.mean(dx[period-1:2*period-1]) if np.sum(~np.isnan(dx[period-1:2*period-1])) > 0 else 0
        for i in range(2*period-1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # Align all to daily timeframe
    camarilla_h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(adx_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Determine market regime based on weekly ADX
        is_trending = adx_1w_aligned[i] >= 25
        is_ranging = adx_1w_aligned[i] < 25
        
        if is_ranging:
            # Ranging market: mean reversion at Camarilla H3/L3
            if close[i] <= camarilla_l3_aligned[i] and volume_spike[i]:
                if position != 1:
                    position = 1
                    signals[i] = position_size
                else:
                    signals[i] = position_size
            elif close[i] >= camarilla_h3_aligned[i] and volume_spike[i]:
                if position != -1:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = -position_size
            # Exit when price returns to midpoint
            elif position == 1 and close[i] >= (camarilla_h3_aligned[i] + camarilla_l3_aligned[i]) / 2:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] <= (camarilla_h3_aligned[i] + camarilla_l3_aligned[i]) / 2:
                position = 0
                signals[i] = 0.0
            elif position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        else:
            # Trending market: breakout of Camarilla H4/L4
            if close[i] > camarilla_h4_aligned[i] and volume_spike[i]:
                if position != 1:
                    position = 1
                    signals[i] = position_size
                else:
                    signals[i] = position_size
            elif close[i] < camarilla_l4_aligned[i] and volume_spike[i]:
                if position != -1:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = -position_size
            # Exit when price returns to Camarilla H3/L3
            elif position == 1 and close[i] <= camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] >= camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_Camilla_Trend_Filter"
timeframe = "1d"
leverage = 1.0