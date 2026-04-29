#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend filter + ATR(14) volatility filter
# Long when price breaks above Donchian upper band AND price > 1d EMA50 AND ATR ratio > 0.8
# Short when price breaks below Donchian lower band AND price < 1d EMA50 AND ATR ratio > 0.8
# Exit when price reverts to Donchian midpoint (mean reversion)
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 20-50 trades/year on 4h timeframe.
# Donchian channels provide objective breakout levels, 1d EMA50 filters counter-trend moves in bear markets,
# ATR volatility filter ensures sufficient momentum behind breakouts. This combination works in both bull and bear regimes.

name = "4h_Donchian20_1dEMA50_ATRFilter_v1"
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d data
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Calculate ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR ratio: current ATR / 50-period ATR average (to filter low volatility environments)
    atr_ma_50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = np.where(atr_ma_50 > 0, atr / atr_ma_50, 0.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50, 14, 50)  # Donchian, EMA50, ATR, ATR MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema50 = ema_50_1d_aligned[i]
        curr_upper = donchian_upper[i]
        curr_lower = donchian_lower[i]
        curr_mid = donchian_mid[i]
        curr_atr_ratio = atr_ratio[i]
        
        # Volatility filter: require sufficient momentum (avoid choppy/low volatility breakouts)
        vol_filter = curr_atr_ratio > 0.8
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price reverts to Donchian midpoint (mean reversion)
            if curr_close <= curr_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reverts to Donchian midpoint (mean reversion)
            if curr_close >= curr_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian upper band AND price > 1d EMA50 AND volatility filter
            if curr_close > curr_upper and curr_close > curr_ema50 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian lower band AND price < 1d EMA50 AND volatility filter
            elif curr_close < curr_lower and curr_close < curr_ema50 and vol_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals