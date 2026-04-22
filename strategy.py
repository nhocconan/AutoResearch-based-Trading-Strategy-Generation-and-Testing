#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout with 1w EMA200 trend filter and volume confirmation.
# Donchian breakouts capture strong momentum moves, while the 1w EMA200 ensures trades align with
# the long-term trend, reducing whipsaws in choppy markets. Volume confirmation adds conviction.
# Designed for low trade frequency (~10-25/year) to minimize fee decay. Works in bull markets
# via long breakouts and in bear markets via short breakdowns, following the 1w trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Donchian calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Donchian channels on 1d
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 200-period EMA on 1d close for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Load 1w data for EMA200 trend filter (higher timeframe)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align indicators to 1d timeframe (waits for 1d bar to close)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = donch_high_aligned[i]
        lower = donch_low_aligned[i]
        ema_1d = ema_200_1d_aligned[i]
        ema_1w = ema_200_1w_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_filter = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above Donchian high + above both EMAs + volume
            if price > upper and price > ema_1d and price > ema_1w and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low + below both EMAs + volume
            elif price < lower and price < ema_1d and price < ema_1w and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: reverse signal or loss of trend
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price breaks below Donchian low or falls below either EMA
                if price < lower or price < ema_1d or price < ema_1w:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price breaks above Donchian high or rises above either EMA
                if price > upper or price > ema_1d or price > ema_1w:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_1dEMA200_1wEMA200_Volume"
timeframe = "1d"
leverage = 1.0