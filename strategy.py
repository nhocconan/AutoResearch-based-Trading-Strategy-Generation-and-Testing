#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian(20) breakout + 1w EMA34 trend + volume confirmation
    # Donchian breakouts capture momentum; weekly EMA filter ensures trading with higher timeframe trend
    # Volume confirmation reduces false breakouts. Works in bull/bear by aligning with weekly trend.
    # Target: 7-25 trades/year per symbol.
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    highest_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian to 1d timeframe (no shift needed as we use completed bar)
    highest_20_aligned = align_htf_to_ltf(prices, df_1d, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_20)
    
    # Calculate 1d ATR(14) for volatility filter
    tr_1d = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if i == 0:
            tr_1d[i] = high_1d[i] - low_1d[i]
        else:
            tr_1d[i] = max(
                high_1d[i] - low_1d[i],
                abs(high_1d[i] - close_1d[i-1]),
                abs(low_1d[i] - close_1d[i-1])
            )
    
    atr14_1d = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        if i == 13:
            atr14_1d[i] = np.mean(tr_1d[i-13:i+1])
        else:
            atr14_1d[i] = (atr14_1d[i-1] * 13 + tr_1d[i]) / 14
    
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 for trend direction
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA34 to 1d timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or 
            np.isnan(atr14_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely low volatility (chop)
        vol_ma_50 = np.full(len(df_1d), np.nan)
        for j in range(49, len(df_1d)):
            if j == 49:
                vol_ma_50[j] = np.mean(atr14_1d[j-49:j+1])
            else:
                vol_ma_50[j] = (vol_ma_50[j-1] * 49 + atr14_1d[j]) / 50
        vol_ma_50_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_50)
        
        if np.isnan(vol_ma_50_aligned[i]):
            signals[i] = 0.0
            continue
            
        vol_ratio = atr14_1d_aligned[i] / vol_ma_50_aligned[i]
        # Trade only when volatility is between 0.3x and 3.0x of 50-period average
        if vol_ratio < 0.3 or vol_ratio > 3.0:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        # Regime: only take trades in direction of weekly trend
        weekly_uptrend = close[i] > ema34_1w_aligned[i]
        weekly_downtrend = close[i] < ema34_1w_aligned[i]
        
        # Donchian breakout signals
        long_breakout = close[i] > highest_20_aligned[i]
        short_breakout = close[i] < lowest_20_aligned[i]
        
        long_entry = long_breakout and volume_confirmed and weekly_uptrend
        short_entry = short_breakout and volume_confirmed and weekly_downtrend
        
        # Exit when price crosses opposite Donchian level or weekly trend changes
        long_exit = close[i] < lowest_20_aligned[i] or not weekly_uptrend
        short_exit = close[i] > highest_20_aligned[i] or not weekly_downtrend
        
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

name = "1d_1w_donchian_breakout_volume_trend_v1"
timeframe = "1d"
leverage = 1.0