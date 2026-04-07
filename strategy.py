#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h KAMA Trend + Volume Spike + Daily Close Filter
# Hypothesis: KAMA adapts to market noise, reducing false signals in ranging markets.
# Combined with volume spikes and daily close above/below KAMA, it captures strong trends
# while avoiding whipsaws. Works in both bull and bear markets by following adaptive trend.
# Target: 25-35 trades/year (100-140 total).

name = "4h_kama_trend_volume_daily_close_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for close filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # KAMA (Adaptive Moving Average) parameters
    er_length = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, prepend=close[0]))
    dir = np.abs(np.subtract(close, np.roll(close, er_length)))
    vol = np.sum(np.lib.stride_tricks.sliding_window_view(change, er_length), axis=1)
    # Handle edge case for vol calculation
    vol_padded = np.concatenate([np.full(er_length-1, np.nan), vol])
    er = np.where(vol_padded != 0, dir / vol_padded, 0)
    
    # Smoothing constants
    sc = np.power(er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1), 2)
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(vol_ma[i]) or 
            i >= len(close_1d) or np.isnan(close_1d[i])):
            signals[i] = 0.0
            continue
        
        # Get daily close (already aligned via index, but we need to map 4h to daily)
        # Since we're on 4h timeframe, we use the most recent daily close
        daily_idx = i // 16  # 16 four-hour bars per day
        if daily_idx >= len(close_1d):
            daily_idx = len(close_1d) - 1
        daily_close = close_1d[daily_idx]
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below KAMA or daily close turns bearish
            if close[i] < kama[i] or daily_close < kama[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses above KAMA or daily close turns bullish
            if close[i] > kama[i] or daily_close > kama[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long entry: price above KAMA and daily close above KAMA
                if close[i] > kama[i] and daily_close > kama[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price below KAMA and daily close below KAMA
                elif close[i] < kama[i] and daily_close < kama[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals