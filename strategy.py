#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h breakout at daily Donchian(20) with 1d volume spike and ADX trend filter
    # Designed for low trade frequency (20-50/year) to minimize fee drag on 4h timeframe
    # Works in bull (breakout continuation) and bear (breakdown continuation) via ADX filter
    # Uses 1d for signal structure (Donchian/volume), 4h only for entry timing precision
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for HTF Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate 1d Donchian channels (20-period)
    donchian_h_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_l_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume average (20-period)
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Get 4h data for ADX trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ADX(14) on 4h
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            elif plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            else:
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        
        return adx
    
    adx_14_4h = calculate_adx(high_4h, low_4h, close_4h, 14)
    
    # Align all HTF indicators to 4h primary timeframe
    donchian_h_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_h_20)
    donchian_l_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_l_20)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    adx_14_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_14_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_h_20_aligned[i]) or 
            np.isnan(donchian_l_20_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(adx_14_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.8x 20-period average
        idx_1d = i // 6  # 6 4h bars per 1d bar
        if idx_1d >= len(volume_1d):
            signals[i] = 0.0
            continue
        volume_confirmed = volume_1d[idx_1d] > 1.8 * vol_avg_20_1d_aligned[i]
        
        # Breakout conditions at 1d Donchian levels
        breakout_long = close[i] > donchian_h_20_aligned[i]  # Price above upper channel -> long
        breakout_short = close[i] < donchian_l_20_aligned[i]  # Price below lower channel -> short
        
        # Trend filter: only trade when ADX > 25 (strong trend)
        trend_filter = adx_14_4h_aligned[i] > 25
        
        # Entry conditions
        enter_long = breakout_long and volume_confirmed and trend_filter
        enter_short = breakout_short and volume_confirmed and trend_filter
        
        # Exit conditions: price returns to opposite Donchian level or ADX weakens
        exit_long = position == 1 and (close[i] < donchian_l_20_aligned[i] or adx_14_4h_aligned[i] < 20)
        exit_short = position == -1 and (close[i] > donchian_h_20_aligned[i] or adx_14_4h_aligned[i] < 20)
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_breakout_volume_adx_v1"
timeframe = "4h"
leverage = 1.0