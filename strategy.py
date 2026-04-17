#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Donchian Channel (20-period) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper band: highest high over 20 periods
    donchian_high = np.full_like(high_12h, np.nan)
    for i in range(len(high_12h)):
        if i >= 19:
            donchian_high[i] = np.max(high_12h[i-19:i+1])
        elif i > 0:
            donchian_high[i] = np.max(high_12h[max(0, i-9):i+1])
        else:
            donchian_high[i] = high_12h[0]
    
    # Lower band: lowest low over 20 periods
    donchian_low = np.full_like(low_12h, np.nan)
    for i in range(len(low_12h)):
        if i >= 19:
            donchian_low[i] = np.min(low_12h[i-19:i+1])
        elif i > 0:
            donchian_low[i] = np.min(low_12h[max(0, i-9):i+1])
        else:
            donchian_low[i] = low_12h[0]
    
    # === 12h ATR (14-period) for volatility filter ===
    tr_12h = np.zeros_like(high_12h)
    tr_12h[0] = high_12h[0] - low_12h[0]
    for i in range(1, len(high_12h)):
        tr_12h[i] = max(
            high_12h[i] - low_12h[i],
            abs(high_12h[i] - high_12h[i-1]),
            abs(low_12h[i] - low_12h[i-1])
        )
    
    atr_12h = np.full_like(tr_12h, np.nan)
    atr_period = 14
    for i in range(len(tr_12h)):
        if i < atr_period:
            if i == 0:
                atr_12h[i] = tr_12h[i]
            else:
                atr_12h[i] = np.mean(tr_12h[:i+1])
        else:
            atr_12h[i] = (atr_12h[i-1] * (atr_period-1) + tr_12h[i]) / atr_period
    
    # === 12h Volume confirmation ===
    volume_12h = df_12h['volume'].values
    vol_ma_20 = np.full_like(volume_12h, np.nan)
    for i in range(len(volume_12h)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume_12h[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume_12h[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume_12h[0]
    
    vol_confirm = volume_12h > vol_ma_20 * 1.5
    
    # === Align indicators to 6h timeframe ===
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    vol_confirm_aligned = align_htf_to_ltf(prices, df_12h, vol_confirm)
    
    # === 6h ADX (14-period) for trend strength ===
    # Calculate +DI and -DI
    up_move = np.diff(high, prepend=high[0])
    down_move = np.diff(low, prepend=low[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # True Range
    tr = np.zeros_like(high)
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - high[i-1]),
            abs(low[i] - low[i-1])
        )
    
    # Smoothed values
    atr_6h = np.full_like(tr, np.nan)
    plus_di_sm = np.full_like(tr, np.nan)
    minus_di_sm = np.full_like(tr, np.nan)
    
    period = 14
    for i in range(len(tr)):
        if i < period:
            if i == 0:
                atr_6h[i] = tr[i]
                plus_di_sm[i] = plus_dm[i]
                minus_di_sm[i] = minus_dm[i]
            else:
                atr_6h[i] = np.mean(tr[:i+1])
                plus_di_sm[i] = np.mean(plus_dm[:i+1])
                minus_di_sm[i] = np.mean(minus_dm[:i+1])
        else:
            atr_6h[i] = (atr_6h[i-1] * (period-1) + tr[i]) / period
            plus_di_sm[i] = (plus_di_sm[i-1] * (period-1) + plus_dm[i]) / period
            minus_di_sm[i] = (minus_di_sm[i-1] * (period-1) + minus_dm[i]) / period
    
    plus_di = 100 * plus_di_sm / atr_6h
    minus_di = 100 * minus_di_sm / atr_6h
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    adx = np.full_like(dx, np.nan)
    for i in range(len(dx)):
        if i < period:
            if i == 0:
                adx[i] = dx[i]
            else:
                adx[i] = np.mean(dx[:i+1])
        else:
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    # === 6h Session filter (08-20 UTC) ===
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_12h_aligned[i]) or 
            np.isnan(vol_confirm_aligned[i]) or 
            np.isnan(adx[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if outside session
        if not session_filter[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        is_trending = adx[i] > 25
        
        # Entry logic: only enter when flat AND volume confirmation
        if position == 0:
            # Long: Close breaks above Donchian high + volume confirmation + trend
            if (close[i] > donchian_high_aligned[i] and 
                vol_confirm_aligned[i] and 
                is_trending):
                signals[i] = 0.25
                position = 1
                continue
            # Short: Close breaks below Donchian low + volume confirmation + trend
            elif (close[i] < donchian_low_aligned[i] and 
                  vol_confirm_aligned[i] and 
                  is_trending):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: ATR-based trailing stop
        elif position == 1:
            # Calculate trailing stop: highest high since entry minus 2 * ATR
            # We'll use a simplified approach: exit if price drops below entry - 2*ATR
            # Since we don't track entry price, use: close < donchian_low_aligned[i] + 0.5 * ATR
            if close[i] < donchian_low_aligned[i] + 0.5 * atr_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # For short: exit if price rises above donchian_high_aligned[i] - 0.5 * ATR
            if close[i] > donchian_high_aligned[i] - 0.5 * atr_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_DonchianBreakout_ATRVolTrendFilter_v1"
timeframe = "6h"
leverage = 1.0