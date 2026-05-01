#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 1w ATR-based volatility regime filter.
# Long when price breaks above Donchian(20) high AND volume > 2.0x 20-bar average AND 1w ATR(14) < 50th percentile (low volatility regime).
# Short when price breaks below Donchian(20) low AND volume > 2.0x 20-bar average AND 1w ATR(14) < 50th percentile.
# Uses discrete sizing 0.25. Target: 75-200 total trades over 4 years (19-50/year).
# Volatility regime filter avoids whipsaws in high volatility markets, improving win rate in both bull and bear conditions.
# Volume confirmation ensures breakouts are supported by participation.
# Primary timeframe: 4h, HTF: 1d for Donchian structure, 1w for volatility regime.

name = "4h_Donchian20_VolumeSpike_LowVolRegime_v1"
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
    
    # Load 1d data ONCE for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian(20) channels from 1d data (previous completed day)
    donchian_high_raw = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low_raw = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_raw)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_raw)
    
    # Load 1w data for volatility regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1w ATR(14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = np.nan  # first bar has no previous close
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w_raw = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 50th percentile (median) of ATR over lookback window for regime classification
    atr_median_raw = pd.Series(atr_1w_raw).rolling(window=50, min_periods=50).median().values
    low_vol_regime_raw = atr_1w_raw < atr_median_raw  # True when ATR < median (low volatility)
    
    # Align volatility regime to 4h timeframe
    low_vol_regime_aligned = align_htf_to_ltf(prices, df_1w, low_vol_regime_raw.astype(float))
    
    # Volume confirmation: current 4h volume > 2.0x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 70  # warmup for Donchian, ATR median, and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(low_vol_regime_aligned[i]) or np.isnan(vol_ma[i]):
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
            
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)  # Volume spike threshold
        low_vol_regime = bool(low_vol_regime_aligned[i])  # Low volatility regime
        
        # Donchian breakout signals
        breakout_up = curr_high > donchian_high_aligned[i]  # break above Donchian high
        breakout_down = curr_low < donchian_low_aligned[i]  # break below Donchian low
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above Donchian high AND volume confirmation AND low volatility regime
            if (breakout_up and 
                volume_confirm and 
                low_vol_regime):
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low AND volume confirmation AND low volatility regime
            elif (breakout_down and 
                  volume_confirm and 
                  low_vol_regime):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below Donchian low (stoploss) OR volatility regime turns high
            if (curr_low < donchian_low_aligned[i] or 
                not low_vol_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high (stoploss) OR volatility regime turns high
            if (curr_high > donchian_high_aligned[i] or 
                not low_vol_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals