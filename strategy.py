#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for ATR-based volatility filter and Donchian channel calculation.
- Entry: Long when price breaks above Donchian upper(20) AND ATR(14) > 1.5 * ATR(50) AND volume > 1.5 * 4h volume MA(20);
         Short when price breaks below Donchian lower(20) AND ATR(14) > 1.5 * ATR(50) AND volume > 1.5 * 4h volume MA(20).
- Exit: Close-based reversal (opposite signal) or volatility filter exit (signal=0 when ATR ratio < 1.2).
- Signal size: 0.30 discrete to balance profit potential and fee drag.
- Donchian channels provide clear breakout levels; ATR expansion filter ensures trades only in volatile regimes; volume confirmation avoids false breakouts.
- Works in bull markets (buy breakouts) and bear markets (sell breakdowns) with volatility filter to avoid choppy periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR and Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) and ATR(50) on 1d
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    tr = np.concatenate([[np.nan], tr2])  # First TR is NaN
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Calculate Donchian channels (20-period) on 1d
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Get 4h data for volume MA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need enough bars for ATR(50) and Donchian(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volatility filter exit: if ATR ratio falls below 1.2, exit position
        if position != 0:
            atr_ratio = atr_14_aligned[i] / atr_50_aligned[i] if atr_50_aligned[i] > 0 else 0
            if atr_ratio < 1.2:
                signals[i] = 0.0
                position = 0
                continue
        
        # Breakout conditions with volatility and volume confirmation
        bullish_breakout = curr_high > donch_high_aligned[i]  # Break above upper channel
        bearish_breakout = curr_low < donch_low_aligned[i]    # Break below lower channel
        
        # Volatility filter: ATR expansion (short-term ATR > 1.5 * long-term ATR)
        vol_expansion = atr_14_aligned[i] > 1.5 * atr_50_aligned[i]
        
        # Volume confirmation
        vol_confirm = curr_volume > 1.5 * vol_ma_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_expansion and vol_confirm:
                # Long: Price breaks above Donchian upper AND volatility expanding
                if bullish_breakout:
                    signals[i] = 0.30
                    position = 1
                # Short: Price breaks below Donchian lower AND volatility expanding
                elif bearish_breakout:
                    signals[i] = -0.30
                    position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.30
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_1dATR_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0