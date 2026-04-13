#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian breakout with 12h volume confirmation and ADX trend filter
    # Long: price breaks above Donchian(20) high + volume > 1.5x 20-period 4h average + ADX > 25
    # Short: price breaks below Donchian(20) low + volume > 1.5x 20-period 4h average + ADX > 25
    # Uses discrete sizing (0.25) to minimize fee drag and ATR-based stoploss
    # Target: 20-50 trades/year to stay within 4h optimal range (80-200 total over 4 years)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    vol_avg_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20_12h)
    
    # Calculate Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ADX(14) for trend filter
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
    )
    tr = np.concatenate([[np.nan], tr])
    
    plus_dm = np.where((high[1:] - high[:-1]) > (high[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((high[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(high[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    atr = np.zeros(n)
    atr[0] = tr[0] if not np.isnan(tr[0]) else 0
    for i in range(1, n):
        if np.isnan(tr[i]):
            atr[i] = atr[i-1]
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    entry_price = np.full(n, np.nan)
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(vol_avg_20_12h_aligned[i]) or
            np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period 12h average
        volume_confirmed = volume[i] > 1.5 * vol_avg_20_12h_aligned[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        trending = adx[i] > 25
        
        # Breakout conditions: price breaks Donchian levels with volume and trend
        breakout_long = (close[i] > donchian_high[i-1]) and volume_confirmed and trending
        breakout_short = (close[i] < donchian_low[i-1]) and volume_confirmed and trending
        
        # Stoploss: 2x ATR below/above entry
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - 2.0 * atr[i]
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + 2.0 * atr[i]
        
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

name = "4h_12h_donchian_volume_adx_v1"
timeframe = "4h"
leverage = 1.0