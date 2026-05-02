#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d EMA50 trend + volume spike + chop regime filter
# Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Donchian(20) provides clear breakout structure from 12h price action
# 1d EMA50 determines trend bias: long when price > EMA50, short when price < EMA50
# Volume spike (2x 20-period average) confirms institutional participation
# Choppiness index regime filter: CHOP(14) > 61.8 = range (mean revert), CHOP < 38.2 = trending (trend follow)
# Works in bull markets via breakouts with trend alignment and bear markets via fade of false breakouts
# Uses 1d as HTF as specified in experiment #117282
# Discrete position sizing: 0.25 (25% of capital) balances exposure and risk

name = "12h_Donchian20_1dEMA50_VolumeSpike_ChopRegime"
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
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 1d EMA50 trend (prior completed 1d bar's EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 12h volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Calculate 12h Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(highest_high - lowest_low) * 14))
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(abs(high - pd.Series(close).shift(1)))
    tr3 = pd.Series(abs(low - pd.Series(close).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values / 
                               (np.log10(highest_high - lowest_low) * 14))
    chop_raw = np.where((highest_high - lowest_low) > 0, chop_raw, 50.0)  # avoid division by zero
    chop_raw = np.nan_to_num(chop_raw, nan=50.0)
    
    chop_regime_trending = chop_raw < 38.2  # trending market
    chop_regime_ranging = chop_raw > 61.8   # ranging market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(chop_regime_trending[i]) or np.isnan(chop_regime_ranging[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high AND price > 1d EMA50 (bullish bias) 
            # AND volume spike AND trending regime (CHOP < 38.2)
            if (close[i] > donchian_high[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i] and 
                chop_regime_trending[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low AND price < 1d EMA50 (bearish bias) 
            # AND volume spike AND trending regime (CHOP < 38.2)
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i] and 
                  chop_regime_trending[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below Donchian low OR below 1d EMA50 (trend change)
            if close[i] < donchian_low[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above Donchian high OR above 1d EMA50 (trend change)
            if close[i] > donchian_high[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals