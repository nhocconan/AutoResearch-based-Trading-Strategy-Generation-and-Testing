#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian channel breakout with weekly ATR regime filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND weekly ATR(14) < 0.08 (low volatility regime) AND volume > 1.5x 20-bar average.
# Short when price breaks below Donchian(20) low AND weekly ATR(14) < 0.08 AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 50-150 total trades over 4 years (12-37/year).
# Weekly ATR filter ensures we only trade breakouts in low volatility regimes (reduces false breakouts in choppy markets).
# Primary timeframe: 6h, HTF: 1w for ATR regime filter.

name = "6h_Donchian20_WeeklyATR_VolumeBreakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for ATR regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate weekly ATR(14) for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range calculation
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Prepend first TR value (high-low of first bar)
    tr = np.concatenate([[high_1w[0] - low_1w[0]], tr])
    
    # ATR(14) using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr_1w = np.zeros_like(tr)
    atr_1w[13] = np.mean(tr[:14])  # First ATR value
    for i in range(14, len(tr)):
        atr_1w[i] = (atr_1w[i-1] * 13 + tr[i]) / 14
    
    # Normalize ATR by price to get percentage
    atr_pct_1w = atr_1w / close_1w
    
    # Regime filter: low volatility when ATR% < 0.08 (8%)
    low_vol_regime = atr_pct_1w < 0.08
    
    # Align weekly ATR regime to 6h timeframe
    low_vol_aligned = align_htf_to_ltf(prices, df_1w, low_vol_regime.astype(float))
    
    # Donchian channel (20-period) on 6h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current 6h volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(low_vol_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        curr_low_vol = low_vol_aligned[i] > 0.5  # Convert to boolean
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)  # Volume spike threshold
        
        # Donchian breakout signals
        breakout_up = curr_high > donchian_high[i]  # break above upper band
        breakout_down = curr_low < donchian_low[i]  # break below lower band
        
        # Entry conditions (only in low volatility regime)
        if position == 0:  # Flat - look for new entries
            # Long: breakout above Donchian high AND low vol regime AND volume confirmation
            if (breakout_up and 
                curr_low_vol and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low AND low vol regime AND volume confirmation
            elif (breakout_down and 
                  curr_low_vol and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below Donchian low (stoploss) OR volatility regime changes
            if (curr_low < donchian_low[i] or 
                not curr_low_vol):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high (stoploss) OR volatility regime changes
            if (curr_high > donchian_high[i] or 
                not curr_low_vol):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals