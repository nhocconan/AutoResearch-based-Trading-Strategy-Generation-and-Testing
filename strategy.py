#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h ATR-based breakout with 1d trend filter and volume confirmation
# Long when price breaks above ATR(14) upper band AND 1d EMA(50) trend up AND volume > 1.5x avg
# Short when price breaks below ATR(14) lower band AND 1d EMA(50) trend down AND volume > 1.5x avg
# Exit when price crosses back through ATR midpoint
# Target: 75-200 total trades over 4 years (19-50/year) for optimal 4h performance

name = "4h_atr_breakout_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR(14) for volatility bands
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # ATR bands (ATR multiplier = 2.0)
    atr_mult = 2.0
    atr_upper = close + atr * atr_mult
    atr_lower = close - atr * atr_mult
    atr_mid = close  # midpoint for exit
    
    # 1d EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    one_day_close = df_1d['close'].values
    one_day_close_series = pd.Series(one_day_close)
    one_day_ema = one_day_close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    one_day_ema_aligned = align_htf_to_ltf(prices, df_1d, one_day_ema)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(one_day_ema_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price crosses back through ATR midpoint
        if position == 1:  # long position
            if close[i] <= atr_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= atr_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend filter and volume confirmation
            # Long: price > ATR upper band AND 1d EMA trending up AND volume confirmation
            if (close[i] > atr_upper[i] and one_day_ema_aligned[i] > one_day_ema_aligned[max(i-1,0)] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < ATR lower band AND 1d EMA trending down AND volume confirmation
            elif (close[i] < atr_lower[i] and one_day_ema_aligned[i] < one_day_ema_aligned[max(i-1,0)] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>