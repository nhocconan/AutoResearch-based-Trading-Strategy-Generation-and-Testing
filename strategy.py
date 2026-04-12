#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    """
    Hypothesis: 1h Donchian(20) breakout + 4h ADX(14) > 25 trend filter + 1d volume spike (volume > 1.5 * 20-period MA)
    - In strong 4h trends (ADX > 25), trade breakouts of 1h Donchian channels in trend direction
    - Requires 1d volume confirmation to avoid low-liquidity false breakouts
    - Uses session filter (08-20 UTC) to reduce noise
    - Discrete sizing 0.20 to minimize fee churn
    - Target: 15-37 trades/year (60-150 over 4 years)
    """
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for ADX trend filter (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h ADX(14) with proper min_periods
    def calculate_adx(high, low, close, period=14):
        n = len(high)
        if n < period + 1:
            return np.full(n, np.nan)
        
        tr = np.zeros(n)
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        
        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
            
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
        
        # Wilder's smoothing (equivalent to EMA with alpha=1/period)
        atr = np.zeros(n)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, n):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros(n)
        minus_di = np.zeros(n)
        dx = np.zeros(n)
        
        for i in range(period, n):
            if atr[i] > 0:
                plus_di[i] = 100 * (np.mean(plus_dm[i-period+1:i+1]) / atr[i])
                minus_di[i] = 100 * (np.mean(minus_dm[i-period+1:i+1]) / atr[i])
                if plus_di[i] + minus_di[i] > 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros(n)
        adx[:2*period-1] = np.nan
        if 2*period-1 < n:
            adx[2*period-1] = np.mean(dx[period:2*period])
            for i in range(2*period, n):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_4h = calculate_adx(high_4h, low_4h, close_4h, 14)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    strong_trend = adx_4h_aligned > 25  # 4h ADX > 25 indicates strong trend
    
    # Get 1d data for volume filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    # Calculate 1d volume 20-period MA
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * vol_ma_1d)  # Volume > 1.5x 20-day MA
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Calculate 1h Donchian channels (20-period)
    def calculate_donchian(high, low, period=20):
        n = len(high)
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        for i in range(period-1, n):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start loop after warmup period (max of all indicators)
    start_idx = max(50, 20)  # ADX needs ~2*14, Donchian needs 20
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position == 1:
                signals[i] = 0.0  # Close long at session end
                position = 0
            elif position == -1:
                signals[i] = 0.0  # Close short at session end
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(strong_trend[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i])):
            signals[i] = 0.0
            continue
        
        # Check for breakout signals
        long_breakout = close[i] > donchian_upper[i]
        short_breakout = close[i] < donchian_lower[i]
        
        # Entry logic: breakout in direction of 4h trend + volume confirmation
        long_entry = long_breakout and strong_trend[i] and volume_spike_aligned[i]
        short_entry = short_breakout and strong_trend[i] and volume_spike_aligned[i]
        
        # Exit logic: opposite breakout or session end
        long_exit = short_breakout or not in_session[i]
        short_exit = long_breakout or not in_session[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_donchian_adx_volume_v1"
timeframe = "1h"
leverage = 1.0