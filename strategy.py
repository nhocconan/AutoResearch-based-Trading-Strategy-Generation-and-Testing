#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volume confirmation and regime filter
# Uses Donchian channel breakout from prior 4h bar for entry
# 1d ATR(14) > 1.5x 50-period MA as volatility expansion filter (avoids chop)
# Volume spike (2.0x 20-period MA) confirms institutional participation
# Works in bull/bear via volatility filter - only trades during high volatility regimes
# Designed for low frequency (75-200 trades over 4 years) to minimize fee drag on 4h timeframe

name = "4h_Donchian20_1dATR_VolumeSpike_VolatilityFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for ATR and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d ATR(14) calculation (volatility filter)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with indices
    
    # ATR using Wilder's smoothing
    def wilders_smoothing(x, period):
        result = np.full_like(x, np.nan)
        if len(x) >= period:
            first_val = np.nansum(x[1:period+1])  # skip first NaN
            result[period] = first_val
            for i in range(period+1, len(x)):
                result[i] = result[i-1] - (result[i-1] / period) + x[i]
        return result
    
    atr_period = 14
    atr = wilders_smoothing(tr, atr_period)
    
    # 1d ATR 50-period moving average
    atr_ma_50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ma_50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50)
    
    # 1d ATR aligned to 4h
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Volatility filter: current ATR > 1.5 * 50-period ATR MA (expanding volatility)
    volatility_expanding = atr_aligned > (atr_ma_50_aligned * 1.5)
    
    # Volume confirmation: 1d volume > 2.0 * 20-period average volume
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    volume_spike = volume_1d_aligned > (volume_ma_20_1d_aligned * 2.0)
    
    # Calculate Donchian levels from prior 4h bar (using prior bar's HH/LL)
    # Donchian(20) = highest high/lowest low of last 20 periods
    def donchian_channel(high_arr, low_arr, period):
        highest_high = pd.Series(high_arr).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low_arr).rolling(window=period, min_periods=period).min().values
        return highest_high, lowest_low
    
    # Use prior bar's data to avoid look-ahead
    prior_high = np.concatenate([[np.nan], high[:-1]])
    prior_low = np.concatenate([[np.nan], low[:-1]])
    
    donchian_high, donchian_low = donchian_channel(prior_high, prior_low, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, 20)  # Need ATR MA50 and Donchian20
    
    for i in range(start_idx, n):
        if (np.isnan(volatility_expanding[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_long = close[i] > donchian_high[i]  # Price breaks above Donchian high
        breakout_short = close[i] < donchian_low[i]  # Price breaks below Donchian low
        
        # Combined filters
        vol_ok = volatility_expanding[i]
        vol_spike_ok = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above Donchian high with volatility expansion and volume spike
            if breakout_long and vol_ok and vol_spike_ok:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below Donchian low with volatility expansion and volume spike
            elif breakout_short and vol_ok and vol_spike_ok:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on close below Donchian low or volatility contraction (<1.0x ATR MA)
            if close[i] < donchian_low[i] or atr_aligned[i] < (atr_ma_50_aligned[i] * 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on close above Donchian high or volatility contraction (<1.0x ATR MA)
            if close[i] > donchian_high[i] or atr_aligned[i] < (atr_ma_50_aligned[i] * 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals