#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA(34) trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND 1d EMA(34) is rising AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian(20) low AND 1d EMA(34) is falling AND volume > 1.5x 20-period average.
# Uses discrete position size 0.25. Donchian captures breakouts in trending markets, 1d EMA ensures higher timeframe trend alignment (avoiding counter-trend trades),
# volume spike confirms institutional participation. Designed to work in both bull (buy breakouts) and bear (sell breakdowns) markets.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while capturing significant moves.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Donchian Channel (20) ===
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = high_ma.values
    donchian_low = low_ma.values
    
    # === 12h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 1d data once before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:  # Need enough for EMA calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: EMA(34) for trend filter ===
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean()
    ema_34_1d_values = ema_34_1d.values
    
    # EMA slope: rising if current > previous, falling if current < previous
    ema_slope = np.diff(ema_34_1d_values, prepend=ema_34_1d_values[0])
    ema_rising = ema_slope > 0
    ema_falling = ema_slope < 0
    
    # Align 1d EMA slope to 12h timeframe
    ema_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_falling)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 34 periods needed for EMA, 20 for Donchian/volume MA)
    warmup = 40
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Donchian low or volume spike ends
            if price <= donchian_low[i] or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Donchian high or volume spike ends
            if price >= donchian_high[i] or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian high AND 1d EMA rising AND volume spike
            if price > donchian_high[i] and ema_rising_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Donchian low AND 1d EMA falling AND volume spike
            elif price < donchian_low[i] and ema_falling_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_Donchian20_1dEMA34_VolumeSpike_V1"
timeframe = "12h"
leverage = 1.0