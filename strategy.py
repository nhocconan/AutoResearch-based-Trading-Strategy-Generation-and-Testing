#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and ATR-based volume confirmation
# Long when price breaks above Donchian upper band AND price > 1d EMA34 AND volume > 1.5x ATR(14) scaled average
# Short when price breaks below Donchian lower band AND price < 1d EMA34 AND volume > 1.5x ATR(14) scaled average
# Exit when price retests Donchian midpoint (mean reversion structure)
# Uses discrete position sizing (0.25) to reduce fee drag and improve test generalization.
# Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years) to avoid overtrading.
# Combines price channel breakout (Donchian) with HTF trend filter (1d EMA34) and volatility-adjusted volume confirmation
# to capture high-probability trends while filtering choppy markets. Works in bull markets by capturing breakouts
# and in bear markets by shorting breakdowns with trend alignment preventing counter-trend trades.

name = "4h_Donchian20_EMA34_VolumeATR_v1"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Calculate ATR(14) for volume confirmation scaling
    tr1 = pd.Series(high).shift(1) - pd.Series(low).shift(1)
    tr2 = abs(pd.Series(high).shift(1) - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low).shift(1) - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: >1.5x ATR(14) scaled average volume (adaptive to volatility)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    vol_atr_scaled = volume_ma_20 * (1.0 + atr_14 / (close + 1e-10))  # Scale by volatility
    volume_confirm = volume > 1.5 * vol_atr_scaled
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 34, 20)  # Donchian, EMA34, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_highest = highest_high[i]
        curr_lowest = lowest_low[i]
        curr_mid = donchian_mid[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price retests Donchian midpoint
            if curr_close <= curr_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retests Donchian midpoint
            if curr_close >= curr_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian upper band AND price > 1d EMA34 AND volume confirmation
            if curr_close > curr_highest and curr_close > curr_ema34_1d and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian lower band AND price < 1d EMA34 AND volume confirmation
            elif curr_close < curr_lowest and curr_close < curr_ema34_1d and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals