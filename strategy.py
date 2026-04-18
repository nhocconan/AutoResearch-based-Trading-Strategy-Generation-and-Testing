#!/usr/bin/env python3
"""
6h_PriceChannel_KeltnerBreakout_With_TrendAndVolume
Hypothesis: Breakouts from Keltner Channels (KC) on 6h with 1d trend filter and volume spike.
In bull markets: buy when price > KC upper band + volume spike + 1d uptrend.
In bear markets: sell when price < KC lower band + volume spike + 1d downtrend.
Uses ATR-based bands to adapt to volatility, reducing false breakouts in ranging markets.
Designed for low trade frequency (12-37/year) to minimize fee drag while capturing
sustained moves in both bull and bear regimes via trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Keltner Channel on 6h: EMA(20) +/- 2*ATR(10)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    kc_upper = ema_20 + 2 * atr_10
    kc_lower = ema_20 - 2 * atr_10
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)  # Warmup for EMA(34) and EMA(20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(kc_upper[i]) or
            np.isnan(kc_lower[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema34_1d = ema_34_1d_aligned[i]
        kc_up = kc_upper[i]
        kc_low = kc_lower[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above KC upper + volume spike + 1d uptrend
            if price > kc_up and vol_spike and price > ema34_1d:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below KC lower + volume spike + 1d downtrend
            elif price < kc_low and vol_spike and price < ema34_1d:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price re-enters KC or trend turns down
            if price < kc_up or price < ema34_1d:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price re-enters KC or trend turns up
            if price > kc_low or price > ema34_1d:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_PriceChannel_KeltnerBreakout_With_TrendAndVolume"
timeframe = "6h"
leverage = 1.0