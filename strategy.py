#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility filter and volume confirmation.
# Long when price breaks above 20-period high AND 1d ATR(14) > 1.5x 50-period median ATR AND volume > 2x 20-bar average.
# Short when price breaks below 20-period low AND 1d ATR(14) > 1.5x 50-period median ATR AND volume > 2x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 75-200 total trades over 4 years (19-50/year).
# Volume spike threshold set to 2.0x to reduce false breakouts and improve signal quality.
# ATR filter ensures we only trade during sufficient volatility regimes, avoiding choppy markets.
# Works in bull markets (trend continuation) and bear markets (volatility expansion breakouts).
# Primary timeframe: 4h, HTF: 1d for ATR filter.

name = "4h_Donchian20_1dATR_Volume_Regime_v2"
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
    
    # Load 1d data ONCE before loop for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for ATR calculation and median
        return np.zeros(n)
    
    # 1d ATR(14) calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar TR
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # First bar TR
    tr3[0] = np.abs(low_1d[0] - close_1d[0])  # First bar TR
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) - Wilder's smoothing
    atr_14 = np.zeros_like(tr)
    atr_14[13] = np.mean(tr[1:14])  # First ATR value
    for i in range(14, len(tr)):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # 50-period median of ATR for regime filter
    atr_median_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).median().values
    
    # Align ATR and median ATR to 4h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_median_aligned = align_htf_to_ltf(prices, df_1d, atr_median_50)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current 4h volume > 2.0x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(atr_aligned[i]) or np.isnan(atr_median_aligned[i]) or \
           np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or np.isnan(vol_ma[i]):
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
            
        # Volatility regime filter: ATR > 1.5x median ATR (ensures sufficient volatility)
        vol_regime = atr_aligned[i] > (1.5 * atr_median_aligned[i])
        
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)  # Volume spike threshold
        
        # Donchian breakout signals
        breakout_up = curr_high > high_ma[i]  # break above 20-period high
        breakout_down = curr_low < low_ma[i]  # break below 20-period low
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above Donchian high AND volatility regime AND volume confirmation
            if (breakout_up and 
                vol_regime and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low AND volatility regime AND volume confirmation
            elif (breakout_down and 
                  vol_regime and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below Donchian low (stoploss) OR volatility regime fails
            if (curr_low < low_ma[i] or 
                not vol_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high (stoploss) OR volatility regime fails
            if (curr_high > high_ma[i] or 
                not vol_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals