#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 1w ADX regime filter
# - Entry: Long when price breaks above 4h Donchian H20 + 1d volume > 1.8x 20-period average + 1w ADX > 25 (trending regime)
#          Short when price breaks below 4h Donchian L20 + 1d volume > 1.8x 20-period average + 1w ADX > 25 (trending regime)
# - Exit: Close-based reversal - exit long when price < 4h Donchian L20, exit short when price > 4h Donchian H20
# - Stoploss: ATR-based - exit when price moves against position by 2.5 * ATR(14)
# - Position sizing: 0.30 (discrete level)
# - Uses 4h price structure for entries/exits, daily volume for participation confirmation,
#   and weekly ADX to filter for trending markets where breakouts work best
# - Target: 75-200 total trades over 4 years (19-50/year) to stay within HARD MAX: 400 total
# - Donchian breakouts capture momentum, volume confirms institutional participation,
#   ADX>25 ensures we only trade in trending markets reducing false breakouts

name = "4h_1d_1w_donchian_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Pre-compute 1d data for volume
    volume_1d = df_1d['volume'].values
    
    # Pre-compute 1w data for ADX
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_h20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_l20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w ADX (14-period)
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.diff(high_1w, prepend=high_1w[0])
    down_move = -np.diff(low_1w, prepend=low_1w[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM
    alpha = 1.0 / 14
    atr_1w = np.zeros_like(tr)
    plus_dm_1w = np.zeros_like(plus_dm)
    minus_dm_1w = np.zeros_like(minus_dm)
    
    atr_1w[0] = tr[0]
    plus_dm_1w[0] = plus_dm[0]
    minus_dm_1w[0] = minus_dm[0]
    
    for i in range(1, len(tr)):
        atr_1w[i] = (1 - alpha) * atr_1w[i-1] + alpha * tr[i]
        plus_dm_1w[i] = (1 - alpha) * plus_dm_1w[i-1] + alpha * plus_dm[i]
        minus_dm_1w[i] = (1 - alpha) * minus_dm_1w[i-1] + alpha * minus_dm[i]
    
    # Avoid division by zero
    plus_di_1w = np.where(atr_1w > 0, 100 * plus_dm_1w / atr_1w, 0.0)
    minus_di_1w = np.where(atr_1w > 0, 100 * minus_dm_1w / atr_1w, 0.0)
    
    dx_1w = np.where((plus_di_1w + minus_di_1w) > 0, 
                     100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w), 
                     0.0)
    
    # Smoothed DX to get ADX
    adx_1w = np.zeros_like(dx_1w)
    adx_1w[13] = np.mean(dx_1w[14:28]) if len(dx_1w) >= 28 else np.mean(dx_1w[14:]) if len(dx_1w) > 14 else 0.0
    for i in range(14, len(dx_1w)):
        adx_1w[i] = (1 - alpha) * adx_1w[i-1] + alpha * dx_1w[i]
    
    # Calculate 4h ATR (14-period) for stoploss
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr1_4h[0] = 0
    tr2_4h[0] = 0
    tr3_4h[0] = 0
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    # Align all HTF data to 4h timeframe
    donchian_h20_aligned = align_htf_to_ltf(prices, prices, donchian_h20)  # 4h data already aligned
    donchian_l20_aligned = align_htf_to_ltf(prices, prices, donchian_l20)  # 4h data already aligned
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    atr_4h_aligned = align_htf_to_ltf(prices, prices, atr_4h)  # 4h data already aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(donchian_h20_aligned[i]) or np.isnan(donchian_l20_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(atr_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 4h close
        close_price = close_4h[i]
        
        # Get current 1d volume for confirmation
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirmation = volume_1d_aligned[i] > 1.8 * volume_ma_aligned[i]
        
        # ADX filter: > 25 indicates trending market (good for breakouts)
        adx_filter = adx_aligned[i] > 25.0
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian H20 + volume confirmation + trending market
            if (close_price > donchian_h20_aligned[i] and 
                volume_confirmation and 
                adx_filter):
                position = 1
                entry_price = close_price
                signals[i] = 0.30
            # Short entry: price breaks below Donchian L20 + volume confirmation + trending market
            elif (close_price < donchian_l20_aligned[i] and 
                  volume_confirmation and 
                  adx_filter):
                position = -1
                entry_price = close_price
                signals[i] = -0.30
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or stoploss
            # Calculate stoploss level
            if position == 1:  # Long position
                stop_loss = entry_price - 2.5 * atr_4h_aligned[i]
                # Exit conditions: price < Donchian L20 OR stoploss hit
                if close_price < donchian_l20_aligned[i] or close_price <= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.30
            else:  # position == -1, Short position
                stop_loss = entry_price + 2.5 * atr_4h_aligned[i]
                # Exit conditions: price > Donchian H20 OR stoploss hit
                if close_price > donchian_h20_aligned[i] or close_price >= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.30
    
    return signals