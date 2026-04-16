#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA(34) trend filter + volume confirmation.
# Long when price breaks above Donchian upper band AND 1d EMA(34) is rising AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian lower band AND 1d EMA(34) is falling AND volume > 1.5x 20-period average.
# Uses discrete position size 0.25. Donchian captures structural breaks, 1d EMA ensures alignment with higher timeframe trend,
# volume spike confirms institutional participation. Designed to work in both bull (breakouts up) and bear (breakdowns down) markets.
# Target: 80-180 trades over 4 years (20-45/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian(20) ===
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min()
    upper_band = high_ma.values
    lower_band = low_ma.values
    
    # === 4h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 1d data once before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:  # Need enough for EMA calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: EMA(34) for trend filter ===
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean()
    ema_34_values = ema_34.values
    
    # Align 1d EMA to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_values)
    
    # Calculate EMA slope (rising/falling)
    ema_slope = np.diff(ema_34_aligned, prepend=ema_34_aligned[0])
    ema_rising = ema_slope > 0
    ema_falling = ema_slope < 0
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 34 periods needed for EMA, 20 for Donchian/volume MA)
    warmup = 40
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to mid-band or volume spike ends
            mid_band = (upper_band[i] + lower_band[i]) / 2
            if price <= mid_band or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to mid-band or volume spike ends
            mid_band = (upper_band[i] + lower_band[i]) / 2
            if price >= mid_band or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price > upper band AND EMA rising AND volume spike
            if price > upper_band[i] and ema_rising[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price < lower band AND EMA falling AND volume spike
            elif price < lower_band[i] and ema_falling[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Donchian20_1dEMA34_VolumeSpike_V1"
timeframe = "4h"
leverage = 1.0