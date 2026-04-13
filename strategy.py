#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with 12h volume confirmation and 1d ADX regime filter
    # Designed to capture strong momentum breakouts in trending markets while avoiding chop
    # ADX > 25 filters for trending regimes; volume confirmation ensures institutional participation
    # Works in both bull and bear markets by trading breakouts in direction of 12h trend
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 12h data for volume confirmation and HTF trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values if 'volume' in df_12h.columns else np.ones(len(df_12h))
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 6h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h volume average (20-period)
    vol_avg_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ADX (14-period) for regime filter
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
            if plus_dm[i] < 0: plus_dm[i] = 0
            if minus_dm[i] < 0: minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        dx = np.abs(plus_di - minus_di) / (np.abs(plus_di) + np.abs(minus_di)) * 100
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        
        # Handle NaN values from insufficient data
        adx[:period] = np.nan
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Calculate 12h EMA20 for trend filter
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all HTF indicators to 6h primary timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, prices, donchian_high)  # 6h data, no alignment needed
    donchian_low_aligned = align_htf_to_ltf(prices, prices, donchian_low)   # 6h data, no alignment needed
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20)
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_avg_20_aligned[i]) or
            np.isnan(ema20_12h_aligned[i]) or
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        volume_confirmed = volume_12h[i // 2] > 1.5 * vol_avg_20_aligned[i] if i // 2 < len(volume_12h) else False
        
        # Breakout conditions at 6h Donchian levels
        breakout_up = close[i] > donchian_high_aligned[i]
        breakout_down = close[i] < donchian_low_aligned[i]
        
        # Trend filter: only trade in direction of 12h EMA20
        trend_filter_long = close[i] > ema20_12h_aligned[i]
        trend_filter_short = close[i] < ema20_12h_aligned[i]
        
        # Regime filter: only trade when ADX > 25 (trending market)
        regime_filter = adx_1d_aligned[i] > 25
        
        # Entry conditions
        enter_long = breakout_up and volume_confirmed and trend_filter_long and regime_filter
        enter_short = breakout_down and volume_confirmed and trend_filter_short and regime_filter
        
        # Exit conditions: opposite Donchian breakout or loss of momentum
        exit_long = position == 1 and (close[i] < donchian_low_aligned[i] or adx_1d_aligned[i] < 20)
        exit_short = position == -1 and (close[i] > donchian_high_aligned[i] or adx_1d_aligned[i] < 20)
        
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

name = "6h_12h_1d_donchian_volume_adx_v1"
timeframe = "6h"
leverage = 1.0