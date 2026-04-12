#!/usr/bin/env python3
"""
4h_1d_Vortex_Trend_v1
Hypothesis: Combine 1-day Vortex Indicator for trend direction with 4-hour price action for entry timing.
Long when VI+ > VI- (bullish trend) and price closes above 4h EMA(21); short when VI- > VI+ (bearish trend) and price closes below 4h EMA(21).
Add volume confirmation: require current volume > 1.5x 20-period average volume.
Uses discrete position sizing (0.25) to limit risk and reduce churn. Targets 20-40 trades/year to minimize fee drag.
Works in bull (follow trend) and bear (follow trend) as Vortex adapts to changing momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Vortex_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Vortex trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily high/low/close for Vortex
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # True Range (TR) for Vortex
    tr = np.maximum(d_high - d_low,
                    np.maximum(np.abs(d_high - np.roll(d_close, 1)),
                               np.absolute(np.abs(d_low - np.roll(d_close, 1)))))
    tr[0] = d_high[0] - d_low[0]  # first TR
    
    # Vortex Indicator components
    vm_plus = np.abs(d_high - np.roll(d_low, 1))
    vm_minus = np.abs(d_low - np.roll(d_high, 1))
    
    # Sum over 14 periods
    tr14 = np.zeros_like(tr)
    vm_plus14 = np.zeros_like(vm_plus)
    vm_minus14 = np.zeros_like(vm_minus)
    
    for i in range(len(tr)):
        if i < 14:
            tr14[i] = np.nan
            vm_plus14[i] = np.nan
            vm_minus14[i] = np.nan
        else:
            tr14[i] = np.sum(tr[i-13:i+1])
            vm_plus14[i] = np.sum(vm_plus[i-13:i+1])
            vm_minus14[i] = np.sum(vm_minus[i-13:i+1])
    
    vi_plus = vm_plus14 / tr14
    vi_minus = vm_minus14 / tr14
    
    # Align Vortex to 4h
    vi_plus_aligned = align_htf_to_ltf(prices, df_1d, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1d, vi_minus)
    
    # 4h EMA(21) for entry filter
    close_s = pd.Series(close)
    ema21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 4h volume average (20-period) for confirmation
    vol_s = pd.Series(volume)
    vol_avg20 = vol_s.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(21, n):
        # Skip if any data invalid
        if (np.isnan(vi_plus_aligned[i]) or np.isnan(vi_minus_aligned[i]) or 
            np.isnan(ema21[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend direction from Vortex
        bullish_trend = vi_plus_aligned[i] > vi_minus_aligned[i]
        bearish_trend = vi_minus_aligned[i] > vi_plus_aligned[i]
        
        # Price relative to EMA
        price_above_ema = close[i] > ema21[i]
        price_below_ema = close[i] < ema21[i]
        
        # Entry logic
        long_entry = bullish_trend and price_above_ema and volume_confirm[i]
        short_entry = bearish_trend and price_below_ema and volume_confirm[i]
        
        # Exit logic: trend reversal
        long_exit = not bullish_trend
        short_exit = not bearish_trend
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
 
EOF