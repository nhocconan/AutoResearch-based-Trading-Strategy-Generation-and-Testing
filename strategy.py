#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime filter
# Elder Ray measures bull/bear power (close - EMA13) filtered by 1d ADX > 25 for trending markets.
# In strong trends (ADX>25), go long when bull power > 0 and rising, short when bear power < 0 and falling.
# Works in both bull and bear markets by only trading in the direction of the 1d trend (ADX) and using Elder Ray for momentum timing.
# Target: 50-150 total trades over 4 years (12-37/year). Size: 0.25.

name = "6h_ElderRay_1dADX25_Regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid TypeError
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for ADX trend filter and EMA13 for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14) for trend strength filter
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
            minus_dm[i] = max(low[i-1] - low[i], 0) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        for i in range(period, len(high)):
            plus_di[i] = (plus_dm[i] * 100) / atr[i] if atr[i] != 0 else 0
            minus_di[i] = (minus_dm[i] * 100) / atr[i] if atr[i] != 0 else 0
            dx[i] = (abs(plus_di[i] - minus_di[i]) * 100) / (plus_di[i] + minus_di[i]) if (plus_di[i] + minus_di[i]) != 0 else 0
        
        adx = np.zeros_like(high)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Calculate 1d EMA13 for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Align 1d indicators to 6h timeframe
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # 6h Elder Ray components: Bull Power = Close - EMA13, Bear Power = EMA13 - Close
    # But we need 6h EMA13 for current power calculation
    ema_13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = close - ema_13_6h
    bear_power = ema_13_6h - close
    
    # 6h volume confirmation: >1.3x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.3 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ADX and EMA need sufficient lookback
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_14_1d_aligned[i]) or
            np.isnan(ema_13_1d_aligned[i]) or
            np.isnan(ema_13_6h[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Skip outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Regime filter: 1d ADX > 25 indicates trending market
        trending_market = adx_14_1d_aligned[i] > 25
        
        # Elder Ray signals (using 6h close and 6h EMA13 for current power)
        # Bull Power rising: current bull power > previous bull power
        bull_power_rising = i > 0 and bull_power[i] > bull_power[i-1]
        # Bear Power falling: current bear power > previous bear power (more negative = stronger bear)
        bear_power_falling = i > 0 and bear_power[i] > bear_power[i-1]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Long entry: trending market + bull power > 0 and rising
        long_entry = trending_market and (bull_power[i] > 0) and bull_power_rising and vol_confirm
        # Short entry: trending market + bear power > 0 and falling (bear power positive = bearish)
        short_entry = trending_market and (bear_power[i] > 0) and bear_power_falling and vol_confirm
        
        # Exit conditions: power weakening or ADX dropping
        long_exit = (bull_power[i] <= 0) or (adx_14_1d_aligned[i] < 20)
        short_exit = (bear_power[i] <= 0) or (adx_14_1d_aligned[i] < 20)
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals