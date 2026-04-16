#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation (1.5x 20-bar avg volume) + ATR-based trend filter (close > EMA50)
# Long when price breaks above Donchian upper band AND volume > 1.5x avg volume AND close > EMA50
# Short when price breaks below Donchian lower band AND volume > 1.5x avg volume AND close < EMA50
# Exit when price crosses EMA50 (trend reversal) or opposite Donchian breakout occurs
# Uses discrete position size 0.25. Volume confirmation reduces false breakouts.
# EMA50 trend filter ensures trades align with medium-term trend to avoid whipsaws in choppy markets.
# 4h timeframe targets 75-200 total trades over 4 years (19-50/year) to minimize fee drag.
# Works in bull markets (capture uptrend breakouts) and bear markets (capture downtrend breakdowns).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian(20), EMA50, Volume MA20 ===
    # Donchian channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # EMA50 for trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Volume MA20 for confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50  # EMA50 needs sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(ema50[i]) or np.isnan(vol_ma20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol = volume[i]
        upper_band = highest_20[i]
        lower_band = lowest_20[i]
        trend = ema50[i]
        vol_ma = vol_ma20[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price < EMA50 (trend break) OR price < lower_band (opposite breakout)
            if (price < trend) or (price < lower_band):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price > EMA50 (trend break) OR price > upper_band (opposite breakout)
            if (price > trend) or (price > upper_band):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume confirmation: current volume > 1.5x 20-bar average
            volume_confirmed = vol > (1.5 * vol_ma)
            
            # LONG: Price breaks above upper band AND volume confirmed AND price > EMA50 (uptrend)
            if (price > upper_band) and volume_confirmed and (price > trend):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below lower band AND volume confirmed AND price < EMA50 (downtrend)
            elif (price < lower_band) and volume_confirmed and (price < trend):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_Donchian20_VolumeConfirmation_EMA50_TrendFilter_V1"
timeframe = "4h"
leverage = 1.0