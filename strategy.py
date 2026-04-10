#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d ATR-based volatility filter and volume confirmation
# - Long when price breaks above Camarilla H3 level (1d) + 1d ATR ratio > 0.8 (normal volatility) + 1d volume > 1.2x 20-period volume SMA
# - Short when price breaks below Camarilla L3 level (1d) + same volatility and volume conditions
# - Exit: close below/above Camarilla L4/H4 levels
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Volatility filter ensures we avoid extremely low volatility periods where breakouts fail
# - Volume confirmation adds conviction to breakouts
# - Target: 30-60 trades/year on 4h timeframe to minimize fee drag while capturing meaningful moves

name = "4h_1d_camarilla_vol_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla levels
    range_1d = high_1d - low_1d
    h3 = pivot + (range_1d * 1.1 / 2.0)  # H3 = pivot + 1.1*(HL)/2
    l3 = pivot - (range_1d * 1.1 / 2.0)  # L3 = pivot - 1.1*(HL)/2
    h4 = pivot + (range_1d * 1.1)        # H4 = pivot + 1.1*(HL)
    l4 = pivot - (range_1d * 1.1)        # L4 = pivot - 1.1*(HL)
    
    # Align 1d Camarilla levels to 4h timeframe (wait for completed 1d bar)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Calculate 1d ATR for volatility filter
    tr1 = np.maximum(high_1d - low_1d, 
                     np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                np.abs(low_1d - np.roll(close_1d, 1))))
    tr1[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR ratio (current ATR / 20-period ATR average) to filter low volatility
    atr_ma_20_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio_1d = np.where(atr_ma_20_1d > 0, atr_1d / atr_ma_20_1d, 1.0)
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Calculate 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    for i in range(30, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(atr_ratio_1d_aligned[i]) or np.isnan(volume_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume for volume spike confirmation
        volume_1d_current = align_htf_to_ltf(prices, df_1d, df_1d['volume'].values)
        
        # Volatility filter: ATR ratio > 0.8 (avoid extremely low volatility)
        vol_filter = atr_ratio_1d_aligned[i] > 0.8
        
        # Volume confirmation: current 1d volume > 1.2x 20-period SMA (moderate volume spike)
        vol_confirm = volume_1d_current[i] > 1.2 * volume_sma_20_1d_aligned[i]
        
        # Camarilla breakout signals
        long_breakout = close[i] > h3_aligned[i]
        short_breakout = close[i] < l3_aligned[i]
        
        # Exit conditions: close beyond extreme Camarilla levels
        exit_long = close[i] < l4_aligned[i]
        exit_short = close[i] > h4_aligned[i]
        
        if position == 0:  # Flat - look for entry
            if long_breakout and vol_filter and vol_confirm:
                position = 1
                signals[i] = 0.25
            elif short_breakout and vol_filter and vol_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals