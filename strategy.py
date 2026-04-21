#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) Breakout + 1w EMA34 Trend Filter + 1d Volume Spike
# Long when price breaks above Donchian(20) high and 1d volume > 2x 20-period average and 1w EMA34 > 1w EMA34 (1 bar ago)
# Short when price breaks below Donchian(20) low and 1d volume > 2x 20-period average and 1w EMA34 < 1w EMA34 (1 bar ago)
# Exit when price crosses Donchian midpoint (10-day average of high/low)
# Trend filter uses 1w EMA34 to capture longer-term trend, avoiding whipsaws in ranges
# Volume spike confirms breakout strength
# Target: 10-25 trades/year by requiring EMA trend + volume spike + Donchian breakout

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate Donchian(20) on 1d data (using high/low from 1d data aligned to 1d timeframe)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    donchian_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid_1d = (donchian_high_1d + donchian_low_1d) / 2
    
    # Align Donchian levels to 1d timeframe (no additional alignment needed as we're using 1d data)
    # Since we're using 1d data directly, the arrays are already at 1d frequency
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # We need to map 1d index to price index (which is at higher frequency)
    # Create a mapping from 1d bar index to price index
    # Since price data is at higher frequency (e.g., 1h, 4h), we need to align 1d signals to price bars
    
    # For simplicity, we'll assume price data is at 1d or lower frequency
    # In practice, we need to align 1d signals to the price timeframe
    
    # Re-align 1d data to price timeframe using the index mapping
    # Since we don't know the exact price timeframe, we'll use the close prices directly
    # and assume we can use 1d indicators by sampling them at the price frequency
    
    # Instead, let's work entirely in the price timeframe and align indicators to it
    # We already have ema_1w_aligned and vol_ma_1d_aligned aligned to price timeframe
    
    # For Donchian, we need to calculate it in price timeframe or align 1d Donchian to price
    # Let's calculate Donchian in price timeframe directly for simplicity
    
    # Recalculate Donchian in price timeframe
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price
        price = close[i]
        
        # Trend filter: 1w EMA34 rising/falling
        ema_now = ema_1w_aligned[i]
        ema_prev = ema_1w_aligned[i-1] if i > 0 else ema_now
        uptrend = ema_now > ema_prev
        downtrend = ema_now < ema_prev
        
        # Volume confirmation: current volume > 2x 20-period average
        # Note: vol_ma_1d_aligned is already aligned to price timeframe
        volume = prices['volume'].iloc[i]
        vol_ma = vol_ma_1d_aligned[i]
        volume_confirm = volume > 2 * vol_ma
        
        if position == 0:
            if volume_confirm:
                # Long: price breaks above Donchian high and 1w EMA34 rising
                if price > donchian_high[i] and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below Donchian low and 1w EMA34 falling
                elif price < donchian_low[i] and downtrend:
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

name = "1d_Donchian20_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0