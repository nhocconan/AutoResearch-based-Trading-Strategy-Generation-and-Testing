#!/usr/bin/env python3
"""
4h_KAMA_Trend_Regime_Volume_V1
Hypothesis: 4h strategy using Kaufman Adaptive Moving Average (KAMA) for trend direction,
filtered by 4h choppiness regime (CHOP < 50 = trending) and volume spike (>1.5x 20-period average).
Enter long when price > KAMA in trending market with volume confirmation.
Enter short when price < KAMA in trending market with volume confirmation.
Exit on ATR(14) trailing stop (2.0*ATR) or opposite KAMA cross.
KAMA adapts to market noise, reducing whipsaws in choppy/ bear markets.
Volume confirmation ensures institutional participation.
Target: 20-35 trades/year (~80-140 total over 4 years) to minimize fee drag.
Works in bull/bear via adaptive trend filter and regime/volume filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for higher timeframe context if needed, but primary is 4h)
    # Note: Strategy primarily uses 4h indicators; HTF load is for template compliance but not used in logic
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 4h Indicators (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Kaufman Adaptive Moving Average (KAMA) - 10/2/30
    # ER = |net change| / sum(|abs change|)
    change = np.abs(np.diff(close_4h, prepend=close_4h[0]))
    net_change = np.abs(np.subtract(close_4h, np.roll(close_4h, 10)))
    net_change[0:10] = 0  # first 10 values invalid
    sum_abs_change = np.convolve(change, np.ones(10), mode='full')[:len(close_4h)]
    sum_abs_change[0:9] = 1  # avoid division by zero
    er = np.where(sum_abs_change > 0, net_change / sum_abs_change, 0)
    # Smoothing constants: fastest SC=2/(2+1)=0.666, slowest SC=2/(30+1)=0.0645
    sc = (er * (0.666 - 0.0645) + 0.0645) ** 2
    kama = np.full_like(close_4h, np.nan)
    kama[9] = close_4h[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close_4h[i] - kama[i-1])
    
    # Choppiness Index (CHOP) - 14 period
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = np.convolve(tr, np.ones(14), mode='full')[:len(tr)]
    atr_sum[0:13] = np.nan  # not enough data
    highest_high = np.convolve(high_4h, np.ones(14), mode='full')[:len(high_4h)]
    highest_high[0:13] = np.nan
    lowest_low = np.convolve(low_4h, np.ones(14), mode='full')[:len(low_4h)]
    lowest_low[0:13] = np.nan
    range_14 = highest_high - lowest_low
    chop = np.where(range_14 > 0, 100 * np.log10(atr_sum / range_14) / np.log10(14), 50)
    
    # Volume spike: >1.5x 20-period average
    vol_ma = np.convolve(volume_4h, np.ones(20), mode='full')[:len(volume_4h)] / 20
    vol_ma[0:19] = np.nan
    volume_spike = volume_4h > (1.5 * vol_ma)
    
    # ATR (14-period) for stoploss
    atr = np.convolve(tr, np.ones(14), mode='full')[:len(tr)] / 14
    atr[0:13] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(kama[i]) or np.isnan(chop[i]) or np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        
        if position == 0:
            # Long conditions: price > KAMA, trending market (CHOP < 50), volume spike
            long_entry = price > kama[i]
            long_regime = chop[i] < 50
            long_volume = volume_spike[i]
            
            # Short conditions: price < KAMA, trending market, volume spike
            short_entry = price < kama[i]
            short_regime = chop[i] < 50
            short_volume = volume_spike[i]
            
            # Entry logic
            if long_entry and long_regime and long_volume:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_entry and short_regime and short_volume:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below KAMA
            elif price < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above KAMA
            elif price > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Trend_Regime_Volume_V1"
timeframe = "4h"
leverage = 1.0