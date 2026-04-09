#!/usr/bin/env python3
# 4h_1d_donchian_breakout_v1
# Hypothesis: Breakout above/below 1d Donchian channel on 4h chart with volume confirmation and ATR stoploss.
# Long when price closes above 1d Donchian high (bullish breakout), short when price closes below 1d Donchian low (bearish breakout).
# Exit when price returns to 1d Donchian middle (mean reversion) or volatility filter fails.
# Works in bull markets by capturing breakouts, in bear markets by fading false breakouts at key levels.
# Target: 20-50 trades/year (80-200 total over 4 years) with strict entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for volatility filter and stoploss
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = 0.9 * atr[i-1] + 0.1 * tr[i]  # Wilder's smoothing
    
    # Load 1d data ONCE before loop for Donchian channel
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d Donchian channel (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    donchian_high = np.zeros(len(df_1d))
    donchian_low = np.zeros(len(df_1d))
    donchian_mid = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        if i < 19:  # Need 20 periods for Donchian
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
            donchian_mid[i] = np.nan
        else:
            donchian_high[i] = np.max(high_1d[i-19:i+1])
            donchian_low[i] = np.min(low_1d[i-19:i+1])
            donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2
    
    # Align Donchian levels to 4h timeframe (wait for previous day's close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    
    # Volume confirmation - 20 period average
    vol_ma_20 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(atr[i]) or np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(donchian_mid_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely high volatility
        vol_filter = atr[i] < 0.06 * close[i]  # ATR less than 6% of price
        
        # Volume confirmation: current volume > 1.25x 20-period average
        vol_ok = volume[i] > vol_ma_20[i] * 1.25
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian middle (mean reversion)
            if close[i] < donchian_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian middle (mean reversion)
            if close[i] > donchian_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above 1d Donchian high with volume confirmation and volatility filter
            if close[i] > donchian_high_aligned[i] and vol_ok and vol_filter:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below 1d Donchian low with volume confirmation and volatility filter
            elif close[i] < donchian_low_aligned[i] and vol_ok and vol_filter:
                position = -1
                signals[i] = -0.25
    
    return signals