#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with HMA(21) trend filter and volume confirmation
# - Uses 1d HTF for HMA(21) to establish primary trend direction
# - Long when price breaks above Donchian upper channel (20-period high) with volume > 1.5x average and 1d HMA up
# - Short when price breaks below Donchian lower channel (20-period low) with volume > 1.5x average and 1d HMA down
# - ATR(14) trailing stop: exit at 2.0x ATR from extreme since entry
# - Fixed position size 0.25 to balance risk and reward
# - Target: 25-40 trades/year on 4h timeframe (100-160 total over 4 years)
# - Works in bull via breakouts, in bear via short breakdowns with trend filter avoiding counter-trend trades

name = "4h_1d_donchian_breakout_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate 1d HMA(21) for trend filter
    close_1d = df_1d['close'].values
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    wma_half = wma(close_1d, half_len)
    wma_full = wma(close_1d, 21)
    wma_2n_minus_n = 2 * wma_half - wma_full
    hma_21 = wma(wma_2n_minus_n, sqrt_len)
    
    # Align 1d HMA to 4h timeframe (wait for completed 1d bar)
    hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21)
    
    # Pre-compute Donchian channels (20-period)
    high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute ATR (14-period) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(hma_21_aligned[i]) or np.isnan(high_roll_max[i]) or 
            np.isnan(low_roll_min[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(atr[i]) or vol_ma_20[i] <= 0 or atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # ATR-based trailing stop: exit if price drops 2.0x ATR from highest high
            if close[i] < highest_high_since_entry - 2.0 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # ATR-based trailing stop: exit if price rises 2.0x ATR from lowest low
            if close[i] > lowest_low_since_entry + 2.0 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Donchian breakout + volume confirmation + 1d HMA trend filter
            if volume_confirmed:
                # Long entry: price breaks above upper Donchian channel AND 1d HMA rising
                if close[i] > high_roll_max[i] and hma_21_aligned[i] > hma_21_aligned[i-1]:
                    position = 1
                    highest_high_since_entry = high[i]
                    lowest_low_since_entry = low[i]
                    signals[i] = 0.25
                # Short entry: price breaks below lower Donchian channel AND 1d HMA falling
                elif close[i] < low_roll_min[i] and hma_21_aligned[i] < hma_21_aligned[i-1]:
                    position = -1
                    highest_high_since_entry = high[i]
                    lowest_low_since_entry = low[i]
                    signals[i] = -0.25
    
    return signals