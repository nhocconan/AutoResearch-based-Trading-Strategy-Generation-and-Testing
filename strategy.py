#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian20 Volume ADX
# Hypothesis: Donchian breakouts with volume confirmation and ADX trend filter capture strong trends
# while avoiding whipsaws in ranging markets. Works in bull via upside breakouts, bear via downside breakouts.
# Volume confirms institutional participation, ADX ensures trending conditions.
# Target: 12-37 trades/year (50-150 total over 4 years) for 12h timeframe.

name = "12h_donchian20_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (ADX)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly ADX(14) for trend strength
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.diff(high)
        minus_dm = -np.diff(low)
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
        
        tr1 = np.abs(np.diff(high))
        tr2 = np.abs(np.diff(low))
        tr3 = np.abs(np.diff(close))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        
        atr = np.zeros_like(close)
        atr[period] = np.mean(tr[:period])
        for i in range(period + 1, len(tr)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        
        plus_di = np.zeros_like(close)
        minus_di = np.zeros_like(close)
        dx = np.zeros_like(close)
        
        for i in range(period, len(plus_dm)):
            plus_di[i] = 100 * (np.mean(plus_dm[i-period+1:i+1]) / atr[i]) if atr[i] != 0 else 0
            minus_di[i] = 100 * (np.mean(minus_dm[i-period+1:i+1]) / atr[i]) if atr[i] != 0 else 0
            dx[i] = (np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) * 100) if (plus_di[i] + minus_di[i]) != 0 else 0
        
        adx = np.zeros_like(close)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(dx)):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Daily Donchian(20) for breakout levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    donchian_high = np.zeros_like(close)
    donchian_low = np.zeros_like(close)
    
    for i in range(20, len(high)):
        donchian_high[i] = np.max(high_1d[max(0, i-20):i])
        donchian_low[i] = np.min(low_1d[max(0, i-20):i])
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_confirm[i]
        
        # Check ADX trend strength (>25 indicates trending market)
        trend_strong = adx_1w_aligned[i] > 25
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or trend weakens
            if close[i] < donchian_low_aligned[i] or not trend_strong:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or trend weakens
            if close[i] > donchian_high_aligned[i] or not trend_strong:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok and trend_strong:
                # Long breakout: price breaks above Donchian high
                if close[i] > donchian_high_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price breaks below Donchian low
                elif close[i] < donchian_low_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals