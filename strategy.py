#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d ATR regime filter + 12h volume spike
# Uses 12h Donchian channels for breakout signals, filtered by 1d ATR-based volatility regime:
#   - High volatility (ATR > 1.5x 20-period EMA ATR) = trend-following mode (breakouts in direction of 1d EMA50)
#   - Low volatility (ATR <= 1.5x EMA ATR) = mean-reversion mode (fade Donchian touches)
# Volume confirmation requires 12h volume > 1.8x 20-period EMA volume
# Designed for 12h timeframe targeting 12-37 trades/year with discrete sizing (0.25)
# Works in bull markets (trend-following breakouts) and bear markets (mean reversion in low vol, trend continuation in high vol)

name = "12h_Donchian20_ATRRegime_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    tr = np.concatenate([[np.nan], tr2])  # first TR is NaN
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d EMA of ATR for volatility regime threshold
    ema_atr_14 = pd.Series(atr_14).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1d indicators to 12h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_atr_14_aligned = align_htf_to_ltf(prices, df_1d, ema_atr_14)
    
    # Get 12h data for Donchian channels (20-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    upper_channel = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper_channel)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower_channel)
    
    # Get 12h data for volume EMA
    vol_12h = df_12h['volume'].values
    vol_ema_20 = pd.Series(vol_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(atr_14_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_atr_14_aligned[i]) or
            np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(vol_ema_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.8 x 20-period EMA
        volume_confirmed = volume[i] > (1.8 * vol_ema_20_aligned[i])
        
        # ATR regime detection
        high_volatility = atr_14_aligned[i] > (1.5 * ema_atr_14_aligned[i])
        
        if position == 0:
            # Determine regime and trade accordingly
            if high_volatility:
                # High volatility: trend-following mode
                # Long: break above upper + volume + price > 1d EMA50
                if (close[i] > upper_aligned[i] and volume_confirmed and 
                    close[i] > ema_50_1d_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: break below lower + volume + price < 1d EMA50
                elif (close[i] < lower_aligned[i] and volume_confirmed and 
                      close[i] < ema_50_1d_aligned[i]):
                    signals[i] = -0.25
                    position = -1
            else:
                # Low volatility: mean-reversion mode
                # Long: touch lower Donchian + volume + price < 1d EMA50 (oversold in downtrend)
                if (close[i] <= lower_aligned[i] and volume_confirmed and 
                    close[i] < ema_50_1d_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: touch upper Donchian + volume + price > 1d EMA50 (overbought in uptrend)
                elif (close[i] >= upper_aligned[i] and volume_confirmed and 
                      close[i] > ema_50_1d_aligned[i]):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price crosses 1d EMA50 in opposite direction OR Donchian reversal
            if (close[i] < ema_50_1d_aligned[i] and high_volatility) or \
               (close[i] > ema_50_1d_aligned[i] and not high_volatility) or \
               close[i] < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses 1d EMA50 in opposite direction OR Donchian reversal
            if (close[i] > ema_50_1d_aligned[i] and high_volatility) or \
               (close[i] < ema_50_1d_aligned[i] and not high_volatility) or \
               close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals