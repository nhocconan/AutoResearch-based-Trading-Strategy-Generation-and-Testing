#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter, volume confirmation, and chop regime filter.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for EMA50 trend filter and chop regime (using ATR-based choppiness index).
- Entry: Long when price breaks above Donchian(20) high AND 1d EMA50 rising AND volume > 1.5 * 4h volume MA(20) AND chop < 61.8 (trending regime).
         Short when price breaks below Donchian(20) low AND 1d EMA50 falling AND volume > 1.5 * 4h volume MA(20) AND chop < 61.8.
- Exit: Opposite Donchian breakout or trend change (EMA50 slope changes sign).
- Signal size: 0.25 discrete to minimize fee drag.
- Donchian provides clear structure; 1d EMA50 ensures intermediate trend alignment; volume avoids false breakouts; chop filter avoids ranging markets.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend) with trend filter to prevent counter-trend trades.
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
    
    # Get 1d data for EMA50 trend and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_slope = ema_50_1d - np.roll(ema_50_1d, 1)
    ema_50_slope[0] = 0
    
    # Calculate 1d ATR(14) for chop regime
    tr1 = np.abs(np.diff(high_1d))
    tr2 = np.abs(np.diff(low_1d))
    tr3 = np.abs(np.diff(close_1d))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.abs(high_1d[0] - low_1d[0])], tr])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d Choppiness Index (CHOP) using ATR and price range
    # CHOP = 100 * log10(sum(ATR14) / (max(high) - min(low))) / log10(14)
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    sum_atr = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    range_hh_ll = max_high - min_low
    chop = np.where(range_hh_ll > 0, 100 * np.log10(sum_atr / range_hh_ll) / np.log10(14), 50)
    chop[np.isnan(chop)] = 50
    
    # Get 4h data for Donchian channels and volume MA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h Donchian(20)
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume MA(20)
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_50_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_50_slope)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for EMA50 and ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_50_slope_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(vol_ma_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Exit conditions
        if position != 0:
            # Exit on trend change (EMA50 slope changes sign)
            if position == 1 and ema_50_slope_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
                continue
            elif position == -1 and ema_50_slope_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
                continue
            # Exit on opposite Donchian breakout
            elif position == 1 and curr_low < donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            elif position == -1 and curr_high > donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions
        bullish_breakout = curr_high > donch_high_aligned[i]  # Break above Donchian high
        bearish_breakout = curr_low < donch_low_aligned[i]    # Break below Donchian low
        
        # Trend filter: only trade in direction of 1d EMA50 slope
        uptrend = ema_50_slope_aligned[i] > 0
        downtrend = ema_50_slope_aligned[i] < 0
        
        # Volume confirmation
        vol_confirm = curr_volume > 1.5 * vol_ma_4h_aligned[i]
        
        # Regime filter: only trade in trending markets (CHOP < 61.8)
        trending_regime = chop_aligned[i] < 61.8
        
        if position == 0:
            # Check for entry signals
            if vol_confirm and trending_regime:
                # Long: Price breaks above Donchian high AND uptrend
                if bullish_breakout and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: Price breaks below Donchian low AND downtrend
                elif bearish_breakout and downtrend:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_EMA50_Trend_Volume_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0