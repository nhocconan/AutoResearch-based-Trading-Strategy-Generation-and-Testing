#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h ADX trend filter with 4h Donchian breakout and volume confirmation
# - Uses 1h ADX(14) > 25 to identify trending markets (works in bull/bear via breakouts)
# - Entry on 1h break of 4h Donchian(20) channels with 4h volume > 1.3x 20-period average
# - Exits via ATR(14) trailing stop (2.5x ATR) or opposite Donchian touch
# - Position size: 0.20 (20% of capital) to control drawdown
# - Target: 15-35 trades/year on 1h (60-140 total over 4 years) to minimize fee drag
# - Uses 4h for signal direction/structure, 1h only for entry timing precision
# - Session filter: 08-20 UTC to avoid low-liquidity Asian session noise

name = "1h_4h_adx_donchian_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h indicators
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # 4h True Range for ATR
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_4h[0] = tr_4h[0]
    
    # 4h ATR(14)
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    # 4h Donchian channels (20-period)
    donchian_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # 4h Volume > 1.3x 20-period average
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_spike_4h = volume_4h > (1.3 * avg_volume_20)
    
    # Align 4h indicators to 1h
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_20)
    volume_spike_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_spike_4h.astype(float))
    
    # Pre-compute 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 1h ADX(14) for trend filter
    tr1_h = high - low
    tr2_h = np.abs(high - np.roll(close, 1))
    tr3_h = np.abs(low - np.roll(close, 1))
    tr_h = np.maximum(tr1_h, np.maximum(tr2_h, tr3_h))
    tr_h[0] = tr_h[0]
    
    atr_1h = pd.Series(tr_h).rolling(window=14, min_periods=14).mean().values
    
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    atr_14_smooth = pd.Series(tr_h).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    plus_di = 100 * (plus_dm_smooth / atr_14_smooth)
    minus_di = 100 * (minus_dm_smooth / atr_14_smooth)
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx_1h = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_trend = adx_1h > 25  # trending market
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_4h_aligned[i]) or np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or np.isnan(volume_spike_4h_aligned[i]) or
            np.isnan(adx_trend[i]) or atr_4h_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, high[i])
            
            # Exit conditions: ATR trailing stop or opposite Donchian touch
            if high[i] >= donchian_high_aligned[i]:  # Touch upper band
                position = 0
                signals[i] = 0.0
            elif low[i] <= highest_since_entry - (2.5 * atr_stop):  # ATR trailing stop
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, low[i])
            
            # Exit conditions: ATR trailing stop or opposite Donchian touch
            if low[i] <= donchian_low_aligned[i]:  # Touch lower band
                position = 0
                signals[i] = 0.0
            elif high[i] >= lowest_since_entry + (2.5 * atr_stop):  # ATR trailing stop
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and trend filter
            if (high[i] >= donchian_high_aligned[i] and  # Break above upper band
                volume_spike_4h_aligned[i] and        # Volume confirmation
                adx_trend[i] and                      # Trending market
                session_filter[i]):                   # Liquid session
                position = 1
                entry_price = high[i]
                atr_stop = atr_4h_aligned[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = 0.20
            elif (low[i] <= donchian_low_aligned[i] and    # Break below lower band
                  volume_spike_4h_aligned[i] and        # Volume confirmation
                  adx_trend[i] and                      # Trending market
                  session_filter[i]):                   # Liquid session
                position = -1
                entry_price = low[i]
                atr_stop = atr_4h_aligned[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -0.20
    
    return signals