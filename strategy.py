#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian channel breakout with 12h volume confirmation and ADX trend filter
    # Long: price breaks above Donchian(20) high + volume > 1.5x 20-period average + ADX > 25
    # Short: price breaks below Donchian(20) low + volume > 1.5x 20-period average + ADX > 25
    # Uses discrete sizing (0.25) to minimize fee drag and ATR-based stoploss
    # Target: 20-50 trades/year to stay within 4h optimal range
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for volume confirmation and ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h volume average (20-period) for confirmation
    vol_avg_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h ADX for trend filter (ADX > 25 indicates strong trend)
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # +DM and -DM
    up_move = np.diff(high_12h)
    down_move = -np.diff(low_12h)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    atr_12h = np.zeros_like(tr_12h)
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    
    for i in range(len(tr_12h)):
        if i < 14:
            if i == 0:
                atr_12h[i] = tr_12h[i] if not np.isnan(tr_12h[i]) else 0.0
                plus_dm_smooth[i] = plus_dm[i]
                minus_dm_smooth[i] = minus_dm[i]
            else:
                prev_atr = atr_12h[i-1] if not np.isnan(atr_12h[i-1]) else 0.0
                prev_plus = plus_dm_smooth[i-1]
                prev_minus = minus_dm_smooth[i-1]
                atr_12h[i] = prev_atr + (tr_12h[i] - prev_atr) / 14 if not np.isnan(tr_12h[i]) else prev_atr
                plus_dm_smooth[i] = prev_plus + (plus_dm[i] - prev_plus) / 14
                minus_dm_smooth[i] = prev_minus + (minus_dm[i] - prev_minus) / 14
        else:
            prev_atr = atr_12h[i-1]
            prev_plus = plus_dm_smooth[i-1]
            prev_minus = minus_dm_smooth[i-1]
            atr_12h[i] = (prev_atr * 13 + tr_12h[i]) / 14
            plus_dm_smooth[i] = (prev_plus * 13 + plus_dm[i]) / 14
            minus_dm_smooth[i] = (prev_minus * 13 + minus_dm[i]) / 14
    
    # DI+ and DI-
    plus_di_12h = np.where(atr_12h != 0, (plus_dm_smooth / atr_12h) * 100, 0)
    minus_di_12h = np.where(atr_12h != 0, (minus_dm_smooth / atr_12h) * 100, 0)
    
    # DX and ADX
    dx_12h = np.where((plus_di_12h + minus_di_12h) != 0, 
                      np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h) * 100, 0)
    
    adx_12h = np.zeros_like(dx_12h)
    for i in range(len(dx_12h)):
        if i < 14:
            adx_12h[i] = np.mean(dx_12h[:i+1]) if i > 0 and not np.isnan(dx_12h[i]) else 0.0
        else:
            adx_12h[i] = (adx_12h[i-1] * 13 + dx_12h[i]) / 14
    
    # Align 12h indicators to 4h timeframe
    vol_avg_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20_12h)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = np.zeros(n)
    donchian_low = np.zeros(n)
    for i in range(n):
        if i < 20:
            donchian_high[i] = np.max(high[:i+1]) if i >= 0 else np.nan
            donchian_low[i] = np.min(low[:i+1]) if i >= 0 else np.nan
        else:
            donchian_high[i] = np.max(high[i-19:i+1])
            donchian_low[i] = np.min(low[i-19:i+1])
    
    # Calculate 4h ATR for stoploss
    atr_4h = np.zeros(n)
    tr_4h = np.zeros(n)
    for i in range(1, n):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        tr_4h[i] = tr
        if i < 14:
            atr_4h[i] = np.mean(tr_4h[1:i+1]) if i > 0 else tr
        else:
            atr_4h[i] = (atr_4h[i-1] * 13 + tr_4h[i]) / 14
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(vol_avg_20_12h_aligned[i]) or
            np.isnan(adx_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average from 12h
        volume_confirmed = volume[i] > 1.5 * vol_avg_20_12h_aligned[i]
        
        # Trend filter: 12h ADX > 25 indicates strong trend
        strong_trend = adx_12h_aligned[i] > 25
        
        # Breakout conditions: price breaks Donchian levels with volume and trend
        breakout_long = (close[i] > donchian_high[i]) and volume_confirmed and strong_trend
        breakout_short = (close[i] < donchian_low[i]) and volume_confirmed and strong_trend
        
        # Stoploss: 2.5x ATR below/above entry
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - 2.5 * atr_4h[i]
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + 2.5 * atr_4h[i]
        
        # Execute signals
        if breakout_long and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            elif position == -1:
                signals[i] = -position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            else:
                signals[i] = 0.0
                entry_price[i] = np.nan
    
    return signals

name = "4h_12h_donchian_volume_adx_v3"
timeframe = "4h"
leverage = 1.0