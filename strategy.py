#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h breakout strategy using 12h Donchian channels with 1d trend filter and volume confirmation.
# Buy when price breaks above 12h Donchian upper (20-period high) and 1d EMA50 > EMA100 (uptrend) with volume spike.
# Sell when price breaks below 12h Donchian lower (20-period low) and 1d EMA50 < EMA100 (downtrend) with volume spike.
# Designed for low trade frequency (~20-50/year) to minimize fee dust. Uses price structure (Donchian) + trend filter + volume.
# Works in bull/bear by only trading with the higher timeframe trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for Donchian channel calculation (once before loop)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 20-period Donchian channel on 12h high/low
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 50 and 100 period EMA on 12h close for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_100_12h = pd.Series(close_12h).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Load 1d data for additional trend confirmation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Align all indicators to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_100_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_100_12h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(ema_100_12h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_100_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        dh = donch_high_aligned[i]
        dl = donch_low_aligned[i]
        ema50_12h = ema_50_12h_aligned[i]
        ema100_12h = ema_100_12h_aligned[i]
        ema50_1d = ema_50_1d_aligned[i]
        ema100_1d = ema_100_1d_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        # Trend filter: EMA50 > EMA100 on both 12h and 1d
        trend_up = (ema50_12h > ema100_12h) and (ema50_1d > ema100_1d)
        trend_down = (ema50_12h < ema100_12h) and (ema50_1d < ema100_1d)
        
        if position == 0:
            # Long conditions: break above Donchian high + uptrend + volume spike
            if price > dh and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below Donchian low + downtrend + volume spike
            elif price < dl and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price returns to Donchian middle or trend breaks
            exit_signal = False
            donch_mid = (dh + dl) / 2
            
            if position == 1:  # long position
                # Exit when price breaks below Donchian middle or trend turns down
                if price < donch_mid or not trend_up:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price breaks above Donchian middle or trend turns up
                if price > donch_mid or not trend_down:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian12h_1dEMA_Trend_Volume"
timeframe = "4h"
leverage = 1.0