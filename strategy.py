#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and ATR volatility filter.
# Long when price breaks above Donchian(20) high AND 1d volume > 1.5x 20-period average AND ATR(14) < ATR(50) (low volatility regime).
# Short when price breaks below Donchian(20) low AND 1d volume > 1.5x 20-period average AND ATR(14) < ATR(50).
# Uses discrete position size 0.25. Donchian breakouts capture momentum, volume confirms participation, low volatility filter reduces whipsaw.
# Designed to work in both bull (buy breakouts) and bear (sell breakdowns) markets by being directionally agnostic to trend.
# Target: 80-120 trades over 4 years (20-30/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Donchian Channel (20-period) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = highest_high
    donchian_low = lowest_low
    
    # === 12h Indicators: ATR (14-period and 50-period for volatility regime) ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    atr_50 = tr.rolling(window=50, min_periods=50).mean().values
    low_volatility = atr_14 < atr_50  # ATR(14) < ATR(50) indicates low volatility regime
    
    # Get 1d data once before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for volume MA calculation
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.5 * vol_ma_1d)
    
    # Align 1d volume spike to 12h timeframe
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for ATR)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr_14[i]) or np.isnan(atr_50[i]) or
            np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        vol_spike = volume_spike_1d_aligned[i]
        low_vol = low_volatility[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price falls below Donchian low (breakdown) or volatility increases
            if price < lower or not low_vol:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price rises above Donchian high (breakout) or volatility increases
            if price > upper or not low_vol:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian high AND volume spike AND low volatility
            if price > upper and vol_spike and low_vol:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Donchian low AND volume spike AND low volatility
            elif price < lower and vol_spike and low_vol:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_Donchian20_1dVolumeSpike_LowVolFilter_V1"
timeframe = "12h"
leverage = 1.0