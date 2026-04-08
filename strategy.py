#!/usr/bin/env python3
# 4h_donchian_breakout_1d_trend_volume_v6
# Hypothesis: 4h Donchian(20) breakout with 1d EMA(50) trend filter and volume confirmation.
# Long when price breaks above 4h Donchian upper band, volume > 1.5x 20-period average, and price above 1d EMA50.
# Short when price breaks below 4h Donchian lower band, volume > 1.5x 20-period average, and price below 1d EMA50.
# Uses volatility filter (ATR ratio < 0.5) to avoid choppy markets. Designed for 20-40 trades/year on 4h.
# Works in bull via breakouts and in bear via short breakdowns with trend alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v6"
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
    
    # 4h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i+1])
        donchian_low[i] = np.min(low[i-20:i+1])
    
    # 4h ATR(14) for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    # ATR ratio (current ATR / 50-period average ATR) for volatility filter
    atr_ma = np.full(n, np.nan)
    for i in range(50, n):
        atr_ma[i] = np.mean(atr[i-50:i+1])
    atr_ratio = np.full(n, np.nan)
    for i in range(50, n):
        if atr_ma[i] > 0:
            atr_ratio[i] = atr[i] / atr_ma[i]
    
    # Volume average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i+1])
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(50, 20)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_ratio[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: avoid choppy markets (ATR ratio < 0.5 = low volatility)
        vol_filter = atr_ratio[i] < 0.5
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian lower band or trend fails
            if close[i] < donchian_low[i] or close[i] < ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper band or trend fails
            if close[i] > donchian_high[i] or close[i] > ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian upper band + volume + trend + vol filter
            if (close[i] > donchian_high[i] and 
                vol_confirm and 
                close[i] > ema50_1d_aligned[i] and 
                vol_filter):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian lower band + volume + trend + vol filter
            elif (close[i] < donchian_low[i] and 
                  vol_confirm and 
                  close[i] < ema50_1d_aligned[i] and 
                  vol_filter):
                position = -1
                signals[i] = -0.25
    
    return signals