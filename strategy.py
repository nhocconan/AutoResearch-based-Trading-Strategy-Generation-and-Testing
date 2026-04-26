#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_RegimeFilter
Hypothesis: On 4h timeframe, Camarilla R1/S1 breakouts with 1d EMA50 trend filter, volume confirmation (>2.0x 20-bar avg), and chop regime filter (CHOP>61.8 = range, CHOP<38.2 = trend) capture institutional breakouts in both bull and bear markets. Uses HTF trend for bias, volume for conviction, and chop filter to avoid whipsaws in sideways markets. Targets 20-50 trades/year to minimize fee drag.
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
    
    # Get 1d data for HTF trend and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA50 on 1d for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels on 1d (using previous day's range)
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # We use the previous completed 1d bar's levels
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = np.nan  # First value has no previous
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    camarilla_range = (prev_high_1d - prev_low_1d) * 1.1 / 12.0
    camarilla_R1 = prev_close_1d + camarilla_range
    camarilla_S1 = prev_close_1d - camarilla_range
    
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Volume average (20-period = ~1.67 days on 4h) for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index on 4h to filter regimes
    # CHOP = 100 * LOG10(SUM(ATR(1),14) / (LOG10(MAX(HIGH,14)-MIN(LOW,14)))) / LOG10(14)
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    atr_period = 14
    tr = np.maximum(high - low, np.absolute(np.roll(high, 1) - np.roll(close, 1)), np.absolute(np.roll(low, 1) - np.roll(close, 1)))
    tr[0] = np.nan  # First value has no previous
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    max_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    min_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Avoid division by zero
    denominator = max_high - min_low
    chop = np.where(denominator != 0, 
                    100 * np.log10(pd.Series(atr).rolling(window=atr_period, min_periods=atr_period).sum().values / denominator) / np.log10(atr_period),
                    50.0)  # Neutral when no range
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(60, 50, 20, atr_period*2)  # 1d lookback, EMA50, volume MA, ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(camarilla_R1_aligned[i]) or 
            np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(chop[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_50_val = ema_50_aligned[i]
        camarilla_R1_val = camarilla_R1_aligned[i]
        camarilla_S1_val = camarilla_S1_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average (stricter to reduce trades)
        volume_confirmed = vol_val > 2.0 * vol_ma_val
        
        # Regime filter: only trade in trending markets (CHOP < 38.2) or strong breaks in ranging markets
        # In ranging markets (CHOP > 61.8), require stronger volume confirmation
        is_ranging = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        # Volume requirement: normal in trending, 2.5x in ranging
        vol_req = 2.5 if is_ranging else 2.0
        volume_confirmed = vol_val > vol_req * vol_ma_val
        
        if position == 0:
            # Long: price breaks above Camarilla R1 with uptrend (close > EMA50) and volume confirmation
            long_signal = (close_val > camarilla_R1_val) and (close_val > ema_50_val) and volume_confirmed
            # Short: price breaks below Camarilla S1 with downtrend (close < EMA50) and volume confirmation
            short_signal = (close_val < camarilla_S1_val) and (close_val < ema_50_val) and volume_confirmed
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Price breaks below Camarilla S1 (strong reversal)
            if close_val < camarilla_S1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: close crosses below EMA50
            elif close_val < ema_50_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 3. Time-based exit: hold max 6 bars (1 day) to avoid overtrading
            # (Not implemented here - relying on price action exits)
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Price breaks above Camarilla R1 (strong reversal)
            if close_val > camarilla_R1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: close crosses above EMA50
            elif close_val > ema_50_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0