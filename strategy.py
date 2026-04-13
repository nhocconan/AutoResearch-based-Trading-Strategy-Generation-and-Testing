#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 1d ATR filter and volume spike confirmation.
    # Long when price breaks above 20-period high AND 1d ATR(14) > 1.5x 50-period MA AND 4h volume > 2.0x 20-period MA.
    # Short when price breaks below 20-period low AND 1d ATR(14) > 1.5x 50-period MA AND 4h volume > 2.0x 20-period MA.
    # Exit when price crosses 10-period EMA in opposite direction.
    # Uses discrete position sizing (0.25) and strict filters to target 75-200 trades over 4 years.
    # Works in bull/bear via volatility filter reducing false breakouts in low-vol regimes.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR(14) 50-period MA for volatility regime filter
    atr_ma_1d = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    
    # Align 1d indicators to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    # Get 4h data for Donchian channels and volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h Donchian(20) channels
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume 20-period MA
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h 10-period EMA for exit signal
    ema_10_4h = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Align 4h indicators to 4h timeframe (no shift needed as we're already in 4h)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    ema_10_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_10_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(atr_ma_1d_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i]) or np.isnan(ema_10_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: 1d ATR(14) > 1.5x 50-period MA (only trade in high volatility regimes)
        volatility_filter = atr_1d_aligned[i] > 1.5 * atr_ma_1d_aligned[i]
        
        # Volume confirmation: current 4h volume > 2.0x 20-period average
        volume_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h)
        volume_spike = volume_4h_aligned[i] > 2.0 * vol_ma_4h_aligned[i]
        
        # Price relative to Donchian channels
        price_above_high = close[i] > donchian_high_aligned[i]
        price_below_low = close[i] < donchian_low_aligned[i]
        
        # EMA-based exit signals
        price_above_ema = close[i] > ema_10_4h_aligned[i]
        price_below_ema = close[i] < ema_10_4h_aligned[i]
        
        # Entry conditions
        if price_above_high and volatility_filter and volume_spike and position != 1:
            position = 1
            signals[i] = position_size
        elif price_below_low and volatility_filter and volume_spike and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions: price crosses 10-period EMA in opposite direction
        elif position == 1 and price_below_ema:
            position = 0
            signals[i] = 0.0
        elif position == -1 and price_above_ema:
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

name = "4h_1d_donchian_breakout_atr_volume_spike_v1"
timeframe = "4h"
leverage = 1.0