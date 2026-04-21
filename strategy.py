#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian(20) breakout with 4h trend filter and volume confirmation
# Long when price breaks above 4h Donchian high AND 4h close > 4h SMA50 AND 1h volume > 1.5x 20-period average
# Short when price breaks below 4h Donchian low AND 4h close < 4h SMA50 AND 1h volume > 1.5x 20-period average
# Exit when price crosses 4h SMA50 in opposite direction
# Uses 4h for direction/trend filter, 1h for entry timing precision
# Session filter: 08-20 UTC to avoid low-volume periods
# Target: 20-35 trades/year by requiring trend alignment + volume spike + breakout

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    vol_4h = df_4h['volume'].values
    
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    sma50 = pd.Series(close_4h).rolling(window=50, min_periods=50).mean().values
    vol_ma = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all 4h indicators to 1h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    sma50_aligned = align_htf_to_ltf(prices, df_4h, sma50)
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma)
    
    # Pre-calculate session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after SMA50 warmup
        # Skip if data not ready or outside session
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(sma50_aligned[i]) or np.isnan(vol_ma_aligned[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        sma50_val = sma50_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume confirmation: current 1h volume > 1.5x 20-period average
        volume_confirm = vol > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: price breaks above Donchian high, above SMA50, volume confirmation
            if price > donch_high_val and price > sma50_val and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Donchian low, below SMA50, volume confirmation
            elif price < donch_low_val and price < sma50_val and volume_confirm:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price crosses below SMA50
                if price < sma50_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price crosses above SMA50
                if price > sma50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Donchian20_Breakout_4hSMA50_Trend_Volume"
timeframe = "1h"
leverage = 1.0