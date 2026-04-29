#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility filter and volume confirmation
# Long when price breaks above Donchian upper channel in high volatility regime (ATR ratio > 1.2) with volume spike
# Short when price breaks below Donchian lower channel in high volatility regime with volume spike
# Uses 1d ATR ratio (current ATR / 20-period ATR average) to filter for explosive breakout conditions
# Volume confirmation ensures breakouts have institutional participation
# Target: 25-35 trades/year (100-140 total over 4 years) to minimize fee drag

name = "4h_Donchian20_1dATRratio1.2_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for daily calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first value NaN
    
    # ATR calculation using Wilder's smoothing
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[1:period+1])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1]/period) + data[i]
        return result
    
    atr_1d = wilders_smooth(tr, 14)
    
    # 20-period ATR average for ratio calculation
    atr_ma_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    
    # ATR ratio: current ATR / 20-period average ATR (>1.2 indicates elevated volatility)
    atr_ratio = np.where(atr_ma_20 > 0, atr_1d / atr_ma_20, 0)
    
    # Align daily ATR ratio to 4h timeframe (completed 1d bar only)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Donchian(20) channels on 4h
    donchian_window = 20
    upper_channel = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(35, 20)  # warmup for ATR ratio and Donchian
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(atr_ratio_aligned[i]) or np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_upper = upper_channel[i]
        curr_lower = lower_channel[i]
        curr_atr_ratio = atr_ratio_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Volatility regime filter: only trade in elevated volatility conditions (ATR ratio > 1.2)
        is_high_vol = curr_atr_ratio > 1.2
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and in high volatility regime
            if is_high_vol and curr_volume_confirm:
                # Bullish breakout: price breaks above upper Donchian channel
                if curr_close > curr_upper:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below lower Donchian channel
                elif curr_close < curr_lower:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price returns to middle of channel OR breaks below lower channel with volume
            middle_channel = (curr_upper + curr_lower) / 2.0
            
            if curr_close <= middle_channel or (curr_close < curr_lower and curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price returns to middle of channel OR breaks above upper channel with volume
            middle_channel = (curr_upper + curr_lower) / 2.0
            
            if curr_close >= middle_channel or (curr_close > curr_upper and curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals