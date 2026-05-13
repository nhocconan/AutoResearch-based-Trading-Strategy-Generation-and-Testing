#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w HMA(21) trend filter and 1d volume confirmation.
# Long when price breaks above Donchian upper band AND price > 1w HMA21 AND 1d volume > 1.5 * 20-period average volume.
# Short when price breaks below Donchian lower band AND price < 1w HMA21 AND 1d volume > 1.5 * 20-period average volume.
# Exit when price crosses Donchian midpoint (mean reversion) or opposite breakout occurs.
# Uses discrete position sizing (0.30) to balance return and drawdown. Designed for BTC/ETH robustness by capturing
# medium-term trends with volume confirmation and weekly trend filter to avoid counter-trend trades.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.

name = "1d_DonchianBreakout_1wHMA21_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w HMA21 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    wma_half = pd.Series(close_1w).ewm(span=half_len, adjust=False, min_periods=half_len).mean().values
    wma_full = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_21_1w = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False, min_periods=sqrt_len).mean().values
    hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    
    # Calculate 1d volume confirmation
    if len(prices) < 20:
        return np.zeros(n)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # Calculate Donchian channels (20-period) on primary timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is NaN
        if (np.isnan(hma_21_1w_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper band AND price > 1w HMA21 AND volume spike
            if (close[i] > highest_high[i] and 
                close[i] > hma_21_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.30
                position = 1
            # SHORT: Price breaks below Donchian lower band AND price < 1w HMA21 AND volume spike
            elif (close[i] < lowest_low[i] and 
                  close[i] < hma_21_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Donchian midpoint OR breaks above upper band (re-entry block)
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price crosses above Donchian midpoint OR breaks below lower band (re-entry block)
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals