#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR filter and volume confirmation.
# Long when price breaks above 4h Donchian upper band AND 1d ATR(14) < median ATR(50) (low volatility regime) AND volume > 1.5x 20-bar average.
# Short when price breaks below 4h Donchian lower band AND 1d ATR(14) < median ATR(50) AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25. Target: 75-200 total trades over 4 years (19-50/year).
# Donchian channels provide clear breakout levels, ATR filter avoids high-chop regimes, volume confirms momentum.
# Primary timeframe: 4h, HTF: 1d for ATR regime filter.

name = "4h_Donchian20_1dATR_Filter_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d ATR(14) for volatility regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first bar has no prior close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Median ATR over 50 days for regime comparison
    median_atr_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).median().values
    
    # Align 1d ATR and median ATR to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    median_atr_50_aligned = align_htf_to_ltf(prices, df_1d, median_atr_50)
    
    # Volume confirmation: current 4h volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Donchian, ATR, and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or \
           np.isnan(atr_14_aligned[i]) or np.isnan(median_atr_50_aligned[i]) or \
           np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)  # Volume spike threshold
        low_vol_regime = atr_14_aligned[i] < median_atr_50_aligned[i]  # Low volatility regime
        
        # Donchian breakout signals
        breakout_up = curr_high > high_20[i]  # break above upper band
        breakout_down = curr_low < low_20[i]  # break below lower band
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above upper band AND low vol regime AND volume confirmation
            if (breakout_up and 
                low_vol_regime and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower band AND low vol regime AND volume confirmation
            elif (breakout_down and 
                  low_vol_regime and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below lower band (stoploss) OR volatility regime changes to high
            if (curr_low < low_20[i] or 
                not low_vol_regime):  # volatility increased
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above upper band (stoploss) OR volatility regime changes to high
            if (curr_high > high_20[i] or 
                not low_vol_regime):  # volatility increased
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals