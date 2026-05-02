#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d EMA50 trend + volume spike + chop regime filter
# Targets 50-150 total trades over 4 years (12-37/year) on 12h timeframe to minimize fee drag
# Donchian(20) provides clear breakout structure on 12h chart
# 1d EMA50 determines long-term trend bias: long when price > EMA50, short when price < EMA50
# Volume spike (2x 20-period average) confirms institutional participation
# Choppiness Index (CHOP) > 61.8 filters out ranging markets, only trade when CHOP < 61.8 (trending)
# Works in bull markets via breakouts with trend alignment and bear markets via fade of false breakouts
# Discrete position sizing: 0.30 (30% of capital) balances exposure and risk

name = "12h_Donchian20_1dEMA50_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian(20) channels (prior completed 12h bar's range)
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 1d EMA50 trend (prior completed 1d bar's EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need at least 50 periods for EMA50
        return np.zeros(n)
    
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 12h volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Calculate 12h Choppiness Index (CHOP) - regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low) / log10(14))
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    # We use CHOP < 61.8 to allow trading in trending markets only
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = 0  # First period has no prior close
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr14.sum() / (highest_high - lowest_low)) / np.log10(14) if (highest_high - lowest_low) != 0 else 50
    # Vectorized CHOP calculation
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh_ll = highest_high - lowest_low
    # Avoid division by zero
    chop = np.where((hh_ll != 0) & (~np.isnan(hh_ll)), 100 * np.log10(atr_sum / hh_ll) / np.log10(14), 50)
    chop_filter = chop < 61.8  # Only trade when not excessively choppy (trending regime)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(20, 50, 14)  # Donchian(20), EMA50, CHOP(14)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(chop_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high AND price > 1d EMA50 (bullish bias) AND volume spike AND trending regime
            if (close[i] > high_ma[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i] and 
                chop_filter[i]):
                signals[i] = 0.30
                position = 1
            # Short entry: price breaks below Donchian low AND price < 1d EMA50 (bearish bias) AND volume spike AND trending regime
            elif (close[i] < low_ma[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i] and 
                  chop_filter[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below Donchian low OR below 1d EMA50 (trend change)
            if close[i] < low_ma[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: price rises above Donchian high OR above 1d EMA50 (trend change)
            if close[i] > high_ma[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals