#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d ADX trend filter and volume confirmation
# Long when: BB width at 20-period low (squeeze) + price breaks above upper BB + 1d ADX > 25 (trending) + volume spike
# Short when: BB width at 20-period low (squeeze) + price breaks below lower BB + 1d ADX > 25 (trending) + volume spike
# Exit when: price returns to middle BB (mean reversion) OR BB width expands > 50% above 20-period average
# Uses 6h timeframe targeting 80-180 total trades over 4 years (20-45/year) to balance opportunity and fee drag
# Works in trending markets (ADX filter) by capturing breakouts from low volatility squeezes

name = "6h_BBSqueeze_1dADX_Trend_VolumeBreakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[1:period])  # first value is simple average
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr_smooth = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
    di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align 1d ADX to 6h timeframe (completed 1d bar only)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 6h Bollinger Bands (20, 2)
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # BB width metrics for squeeze detection
    bb_width_ma_20 = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    bb_width_ratio = bb_width / bb_width_ma_20  # current width relative to 20-period average
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # warmup for ADX and BB calculations
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(adx_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_bb_upper = bb_upper[i]
        curr_bb_lower = bb_lower[i]
        curr_bb_middle = bb_middle[i]
        curr_bb_width_ratio = bb_width_ratio[i]
        curr_adx = adx_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Squeeze condition: BB width < 80% of 20-period average (tight squeeze)
        is_squeeze = curr_bb_width_ratio < 0.8
        
        # Breakout conditions
        breakout_up = curr_close > curr_bb_upper
        breakout_down = curr_close < curr_bb_lower
        
        # Mean reversion exit: return to middle BB
        return_to_middle = (abs(curr_close - curr_bb_middle) < 0.1 * bb_std[i]) if not np.isnan(bb_std[i]) else False
        
        # Expansion exit: BB width > 1.5x 20-period average
        is_expansion = curr_bb_width_ratio > 1.5
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation, squeeze, breakout, and trending regime (ADX > 25)
            if (curr_volume_confirm and is_squeeze and 
                curr_adx > 25):
                # Long breakout
                if breakout_up:
                    signals[i] = 0.25
                    position = 1
                # Short breakout
                elif breakout_down:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: return to middle BB OR BB expansion OR ADX weakens
            if (return_to_middle or is_expansion or curr_adx < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: return to middle BB OR BB expansion OR ADX weakens
            if (return_to_middle or is_expansion or curr_adx < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals