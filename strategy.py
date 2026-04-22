#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ADX(14) trend filter and volume confirmation.
# Long when price breaks above Donchian high(20) in uptrend (ADX>25) with volume spike.
# Short when price breaks below Donchian low(20) in downtrend (ADX>25) with volume spike.
# Exit on opposite Donchian touch or trend reversal (ADX<25).
# Uses 1d ADX for trend strength, Donchian for structure, volume for confirmation.
# Designed for 4h timeframe to target 20-50 trades/year per symbol.
# Works in bull/bear via ADX trend filter + volatility-based entry levels.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for ADX and Donchian levels (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ADX on 1d data
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr_14 = smooth_wilder(tr, 14)
    plus_di_14 = 100 * smooth_wilder(plus_dm, 14) / atr_14
    minus_di_14 = 100 * smooth_wilder(minus_dm, 14) / atr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx_14 = smooth_wilder(dx, 14)
    
    # Calculate Donchian channels (20-period high/low)
    donch_high = np.full_like(high_1d, np.nan)
    donch_low = np.full_like(low_1d, np.nan)
    for i in range(20, len(high_1d)):
        donch_high[i] = np.max(high_1d[i-20:i])
        donch_low[i] = np.min(low_1d[i-20:i])
    
    # Align to 4h timeframe (waits for 1d bar to close)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Higher threshold for fewer trades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(adx_14_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + strong trend (ADX>25) + volume spike
            if (close[i] > donch_high_aligned[i] and 
                adx_14_aligned[i] > 25 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + strong trend (ADX>25) + volume spike
            elif (close[i] < donch_low_aligned[i] and 
                  adx_14_aligned[i] > 25 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit on Donchian low touch or trend weakening (ADX<25)
                if (close[i] < donch_low_aligned[i] or adx_14_aligned[i] < 25):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on Donchian high touch or trend weakening (ADX<25)
                if (close[i] > donch_high_aligned[i] or adx_14_aligned[i] < 25):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dADX14_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0