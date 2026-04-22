#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly EMA200 trend filter and volume confirmation.
# The weekly EMA200 provides a robust long-term trend filter that adapts to both bull and bear markets.
# Daily Donchian(20) breakouts capture intermediate-term momentum, while volume confirmation (>2x 20-day average)
# ensures institutional participation. This combination aims for low trade frequency (~10-25/year) to minimize
# fee decay and works across market regimes by following the higher timeframe trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for EMA200 trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 200-period EMA on weekly close for trend filter
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly EMA200 to daily timeframe (waits for weekly bar to close)
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate daily Donchian channels (20-period high/low)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    vol = prices['volume'].values
    
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-day average volume for volume spike detection
    vol_ma_20 = pd.Series(vol).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_200_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_current = vol[i]
        ema_200 = ema_200_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter: current volume > 2.0 * 20-day average
        vol_spike = vol_current > 2.0 * vol_ma
        
        if position == 0:
            # Long: price breaks above Donchian high + above weekly EMA200 + volume spike
            if price > upper and price > ema_200 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + below weekly EMA200 + volume spike
            elif price < lower and price < ema_200 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price breaks below Donchian low or falls below weekly EMA200
                if price < lower or price < ema_200:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price breaks above Donchian high or rises above weekly EMA200
                if price > upper or price > ema_200:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_1wEMA200_Volume"
timeframe = "1d"
leverage = 1.0