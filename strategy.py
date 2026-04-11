#!/usr/bin/env python3
"""
12h_1d_vortex_breakout_volume_v1
Strategy: 12h Vortex indicator breakout with volume confirmation and 1d trend filter
Timeframe: 12h
Leverage: 1.0
Hypothesis: Uses 12h Vortex Indicator (VI+) and (VI-) crossovers as trend signals, confirmed by volume spike (>1.5x average volume) and filtered by 1d EMA50 trend direction. Designed to capture strong momentum moves in trending markets while avoiding false signals in chop. Works in bull markets (VI+ > VI- with trend) and bear markets (VI- > VI+ with trend). Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_vortex_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h Vortex Indicator (VI+ and VI-)
    tr1 = np.abs(high[1:] - low[:-1])
    tr2 = np.abs(low[1:] - high[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, tr2)])
    vm = np.abs(high - low)
    vi_plus = np.concatenate([[np.nan], np.abs(high[1:] - low[:-1])])
    vi_minus = np.concatenate([[np.nan], np.abs(low[1:] - high[:-1])])
    
    # Smooth with 14-period Wilder's smoothing (EMA with alpha=1/14)
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    vi_plus_smooth = pd.Series(vi_plus).ewm(alpha=1/14, adjust=False).mean().values
    vi_minus_smooth = pd.Series(vi_minus).ewm(alpha=1/14, adjust=False).mean().values
    vi_plus_norm = vi_plus_smooth / atr_14
    vi_minus_norm = vi_minus_smooth / atr_14
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if any required data is invalid
        if (np.isnan(vi_plus_norm[i]) or np.isnan(vi_minus_norm[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        
        # Vortex crossover signals
        vi_cross_up = vi_plus_norm[i] > vi_minus_norm[i] and vi_plus_norm[i-1] <= vi_minus_norm[i-1]
        vi_cross_down = vi_minus_norm[i] > vi_plus_norm[i] and vi_minus_norm[i-1] <= vi_plus_norm[i-1]
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: VI+ crosses above VI- with volume in uptrend
        long_signal = vi_cross_up and vol_confirmed and uptrend_1d
        
        # Short: VI- crosses above VI+ with volume in downtrend
        short_signal = vi_cross_down and vol_confirmed and downtrend_1d
        
        # Exit when Vortex lines re-cross (trend change)
        exit_long = position == 1 and vi_minus_norm[i] > vi_plus_norm[i]
        exit_short = position == -1 and vi_plus_norm[i] > vi_minus_norm[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals