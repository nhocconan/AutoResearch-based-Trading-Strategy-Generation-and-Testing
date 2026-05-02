#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d EMA34 trend + volume spike
# Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Donchian(20) provides clear breakout structure on 12h timeframe
# 1d EMA34 determines medium-term trend bias: long when price > EMA34, short when price < EMA34
# Volume spike (2x 20-period average) confirms institutional participation
# Works in bull markets via breakouts with trend alignment and bear markets via fade of false breakouts
# Discrete position sizing: 0.25 (25% of capital) balances exposure and risk
# Added chop regime filter to reduce whipsaws in ranging markets

name = "12h_Donchian20_1dEMA34_VolumeSpike_Chop"
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
    
    # Calculate 1d EMA34 trend (prior completed 1d bar's EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need at least 34 periods for EMA34
        return np.zeros(n)
    
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 12h volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Calculate 12h Choppiness Index (CHOP) regime filter
    # CHOP > 61.8 = ranging market (mean revert), CHOP < 38.2 = trending (trend follow)
    atr_period = 14
    tr1 = pd.Series(high).rolling(window=2).max() - pd.Series(low).rolling(window=2).min()
    tr2 = abs(pd.Series(high) - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low) - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period, min_periods=atr_period).mean().values
    
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    chop = 100 * np.log10(atr.sum() / (highest_high - lowest_low)) / np.log10(atr_period)
    chop = pd.Series(chop).rolling(window=atr_period, min_periods=atr_period).mean().values
    chop_filter = (chop < 38.2)  # Only trade in trending markets
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(20, 34, atr_period)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(chop_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high AND price > 1d EMA34 (bullish bias) 
            # AND volume spike AND trending market (CHOP < 38.2)
            if (close[i] > high_ma[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_spike[i] and 
                chop_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low AND price < 1d EMA34 (bearish bias) 
            # AND volume spike AND trending market (CHOP < 38.2)
            elif (close[i] < low_ma[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike[i] and 
                  chop_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below Donchian low OR below 1d EMA34 (trend change)
            if close[i] < low_ma[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above Donchian high OR above 1d EMA34 (trend change)
            if close[i] > high_ma[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals