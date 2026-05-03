#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ATR volatility filter and volume confirmation.
# Long when price breaks above 20-period Donchian high + volume spike in low volatility regime.
# Short when price breaks below 20-period Donchian low + volume spike in low volatility regime.
# Uses ATR to filter out high volatility periods where breakouts fail, focusing on low vol breakouts that tend to continue.
# Designed for 50-150 total trades over 4 years (12-37/year) with Sharpe > 0.5 on BTC/ETH/SOL.
# Works in bull via breakout continuation and in bear via short breakdowns with volatility filter.

name = "6h_Donchian20_1dATR_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR(14) using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr_14 = np.full_like(tr, np.nan)
    for i in range(len(tr)):
        if i == 0:
            atr_14[i] = np.nan
        elif i < 14:
            if i == 1:
                atr_14[i] = tr[i]
            else:
                atr_14[i] = (atr_14[i-1] * (i-1) + tr[i]) / i
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Align ATR to 6h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 6h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume regime: current 6h volume > 1.8x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    # Volatility filter: ATR below its 50-period MA (low volatility regime)
    atr_ma_50 = pd.Series(atr_14_aligned).rolling(window=50, min_periods=50).mean().values
    low_volatility = atr_14_aligned < atr_ma_50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_ma_50[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        upper_channel = donchian_high[i]
        lower_channel = donchian_low[i]
        atr_val = atr_14_aligned[i]
        atr_ma_val = atr_ma_50[i]
        vol_spike = volume_spike[i]
        low_vol = low_volatility[i]
        
        # Breakout conditions
        long_breakout = close_val > upper_channel
        short_breakout = close_val < lower_channel
        
        # Entry logic - only in low volatility regime with volume spike
        if position == 0:
            if low_vol and vol_spike and long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif low_vol and vol_spike and short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long exit: price re-enters Donchian channel or volatility spikes
            if close_val < upper_channel:  # Exit when price moves back below upper channel
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters Donchian channel or volatility spikes
            if close_val > lower_channel:  # Exit when price moves back above lower channel
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals