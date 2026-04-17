#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channel (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Donchian channel (20-period)
    highest_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian to 12h
    highest_high_1d_aligned = align_htf_to_ltf(prices, df_1d, highest_high_1d)
    lowest_low_1d_aligned = align_htf_to_ltf(prices, df_1d, lowest_low_1d)
    
    # Calculate daily ATR for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate daily ADX for trend strength filter
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / tr_14
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / tr_14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 12-period volume moving average for confirmation
    volume_ma12 = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need daily Donchian, ATR, ADX, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high_1d_aligned[i]) or 
            np.isnan(lowest_low_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or 
            np.isnan(volume_ma12[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR > 0.5 * 20-period average ATR
        atr_ma20 = pd.Series(atr_1d_aligned).rolling(window=20, min_periods=20).mean().values
        volatility_filter = atr_1d_aligned[i] > (0.5 * atr_ma20[i])
        
        # Trend strength filter: ADX > 20
        trend_filter = adx_1d_aligned[i] > 20
        
        # Volume confirmation: current volume > 1.5 * 12-period average
        volume_filter = volume[i] > (1.5 * volume_ma12[i])
        
        if position == 0:
            # Long: price breaks above daily Donchian high with trend and volume
            if (close[i] > highest_high_1d_aligned[i] and 
                trend_filter and 
                volume_filter and 
                volatility_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily Donchian low with trend and volume
            elif (close[i] < lowest_low_1d_aligned[i] and 
                  trend_filter and 
                  volume_filter and 
                  volatility_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to daily Donchian low
            if close[i] < lowest_low_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to daily Donchian high
            if close[i] > highest_high_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_DailyDonchianBreakout_ADX_Volume"
timeframe = "12h"
leverage = 1.0