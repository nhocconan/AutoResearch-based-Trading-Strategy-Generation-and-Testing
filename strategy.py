#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike_ATRStop
Hypothesis: Donchian channel breakout with 1d EMA50 trend filter and volume spike confirmation on 4h timeframe.
Long when price breaks above 20-period Donchian high with volume > 1.8x average and daily uptrend (close > EMA50).
Short when price breaks below 20-period Donchian low with volume > 1.8x average and daily downtrend (close < EMA50).
Uses discrete sizing 0.28 to balance profit potential and risk. ATR-based stoploss exits when price moves 2.2x ATR against position.
Designed for BTC/ETH with focus on strong trending moves confirmed by higher timeframe trend and volume.
Target trades: 25-40/year (100-160 total over 4 years) to minimize fee drag while capturing significant moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channel (20-period) for breakout signals
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 1.8x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss (using 14-period ATR)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 1d EMA (50), Donchian (20), volume MA (20), ATR (14)
    start_idx = max(50, 20, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(high_ma[i]) or 
            np.isnan(low_ma[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.28
            else:
                signals[i] = -0.28
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        upper_channel = high_ma[i]
        lower_channel = low_ma[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume confirmation and daily uptrend
            long_signal = (high_val > upper_channel) and (volume_val > 1.8 * vol_ma_val) and (close_val > ema_50_1d_val)
            # Short: price breaks below Donchian low with volume confirmation and daily downtrend
            short_signal = (low_val < lower_channel) and (volume_val > 1.8 * vol_ma_val) and (close_val < ema_50_1d_val)
            
            if long_signal:
                signals[i] = 0.28
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.28
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.28
            # Exit: ATR stoploss or trend reversal
            if close_val < entry_price - 2.2 * atr_val or close_val < ema_50_1d_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.28
            # Exit: ATR stoploss or trend reversal
            if close_val > entry_price + 2.2 * atr_val or close_val > ema_50_1d_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0