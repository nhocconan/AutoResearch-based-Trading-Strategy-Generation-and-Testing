#!/usr/bin/env python3
# 4h_volatility_breakout_volume_v3
# Hypothesis: 4h strategy using volatility breakout with volume confirmation and 1d chop regime filter.
# Enters long when price breaks above Donchian(20) high with volume > 1.5x average and chop > 61.8 (trending).
# Enters short when price breaks below Donchian(20) low with volume > 1.5x average and chop > 61.8 (trending).
# Uses discrete sizing (±0.25) to minimize fee churn. Target: 75-200 trades over 4 years.
# Works in bull/bear by using chop regime to avoid false breakouts in ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_volatility_breakout_volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d HTF data for choppiness index (regime filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on daily data
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate choppiness index: CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (HHV - LLV)))
    # Simplified: CHOP = 100 * log10( sum(TR(14)) / log10(14) / (max(high)-min(low)) ) over 14 periods
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_1d = 100 * np.log10(sum_tr_14 / (np.log10(14) * (highest_high_14 - lowest_low_14) + 1e-10))
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # 4h indicators: Donchian(20) and volume average
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(avg_volume_20[i]) or np.isnan(chop_1d_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in trending markets (choppiness > 61.8)
        trending = chop_1d_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price falls below Donchian low OR volume drops below average
            if (close[i] < lowest_low_20[i]) or (volume[i] < 0.5 * avg_volume_20[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above Donchian high OR volume drops below average
            if (close[i] > highest_high_20[i]) or (volume[i] < 0.5 * avg_volume_20[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high with volume spike AND trending regime
            if (close[i] > highest_high_20[i]) and \
               (volume[i] > 1.5 * avg_volume_20[i]) and \
               trending:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low with volume spike AND trending regime
            elif (close[i] < lowest_low_20[i]) and \
                 (volume[i] > 1.5 * avg_volume_20[i]) and \
                 trending:
                position = -1
                signals[i] = -0.25
    
    return signals