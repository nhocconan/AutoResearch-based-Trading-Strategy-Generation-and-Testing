#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Donchian breakouts capture momentum bursts after consolidation periods
# Daily EMA34 ensures we trade breakouts in direction of higher timeframe trend
# Volume confirmation validates breakout strength and reduces false signals
# Works in bull markets (trend continuation) and bear markets (trend resumption after pullbacks)
# Target: 20-50 trades/year (80-200 total over 4 years)

name = "4h_Donchian20_1dEMA34_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for daily calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Donchian Channel (20) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 34, 20, 20)  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_donch_upper = donchian_upper[i]
        curr_donch_lower = donchian_lower[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema34_1d = ema34_1d_aligned[i]
        
        # Determine trend regime from daily EMA34
        bullish_regime = curr_close > curr_ema34_1d
        bearish_regime = curr_close < curr_ema34_1d
        
        if position == 0:  # Flat - look for new entries
            # Look for Donchian breakout with volume confirmation
            if curr_volume_confirm:
                # Bullish breakout: price breaks above upper Donchian in bullish regime
                if bullish_regime and curr_close > curr_donch_upper:
                    signals[i] = 0.30
                    position = 1
                # Bearish breakout: price breaks below lower Donchian in bearish regime
                elif bearish_regime and curr_close < curr_donch_lower:
                    signals[i] = -0.30
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price crosses below lower Donchian OR trend regime changes
            if curr_close < curr_donch_lower or not bullish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price crosses above upper Donchian OR trend regime changes
            if curr_close > curr_donch_upper or not bearish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals