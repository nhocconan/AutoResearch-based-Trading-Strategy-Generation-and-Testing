#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) Breakout + 12h EMA Trend + Volume Spike
# Long when price breaks above Donchian(20) high and 12h EMA(50) rising and volume > 1.5x 20-period average
# Short when price breaks below Donchian(20) low and 12h EMA(50) falling and volume > 1.5x 20-period average
# Exit when price crosses Donchian midpoint
# Uses 12h trend to avoid counter-trend trades, volume spike for confirmation
# Target: 20-40 trades/year by requiring EMA trend + volume spike + Donchian breakout

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA(50)
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_prev = np.roll(ema_50, 1)
    ema_50_prev[0] = ema_50[0]
    ema_rising = ema_50 > ema_50_prev
    
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    ema_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_rising)
    
    # Calculate 12h volume moving average (20-period)
    vol_ma_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Calculate Donchian(20) on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(ema_50_aligned[i]) or np.isnan(ema_rising_aligned[i]) or np.isnan(vol_ma_12h_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ma = vol_ma_12h_aligned[i]
        volume_confirm = volume > 1.5 * vol_ma
        
        # Trend filter: EMA rising for long, falling for short
        ema_rising = ema_rising_aligned[i]
        ema_falling = not ema_rising
        
        if position == 0:
            if volume_confirm:
                # Long: price breaks above Donchian high and EMA rising
                if price > donchian_high[i] and ema_rising:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below Donchian low and EMA falling
                elif price < donchian_low[i] and ema_falling:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit when price crosses Donchian midpoint
            exit_signal = False
            
            if position == 1:  # long position
                if price < donchian_mid[i]:
                    exit_signal = True
            
            elif position == -1:  # short position
                if price > donchian_mid[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0