#!/usr/bin/env python3
# 1d_Weekly_Close_Reversion_With_Volume_Filter
# Hypothesis: Weekly close reversion with volume confirmation exploits weekly mean reversion
# while filtering out low-probability setups. Works in bull/bear by fading extreme weekly closes.
# Weekly close above upper Bollinger Band = short signal, below lower band = long signal.
# Volume confirmation ensures institutional participation. Low trade frequency expected.

name = "1d_Weekly_Close_Reversion_With_Volume_Filter"
timeframe = "1d"
leverage = 1.0

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
    
    # Weekly close data for Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly Bollinger Bands (20-period, 2 std dev)
    period = 20
    std_dev = 2
    close_series = pd.Series(close_1w)
    sma = close_series.rolling(window=period, min_periods=period).mean()
    std = close_series.rolling(window=period, min_periods=period).std()
    upper_band = (sma + std * std_dev).values
    lower_band = (sma - std * std_dev).values
    
    # Weekly volume average (20-period)
    vol_series = pd.Series(volume_1w)
    vol_ma = vol_series.rolling(window=period, min_periods=period).mean().values
    
    # Align weekly data to daily timeframe (wait for weekly bar to close)
    upper_band_aligned = align_htf_to_ltf(prices, df_1w, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1w, lower_band)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = period  # Need enough history for weekly BB
    
    for i in range(start_idx, n):
        if np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or \
           np.isnan(vol_ma_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current weekly volume > 1.5 * 20-week average
        vol_confirm = volume_1w[i] > 1.5 * vol_ma_aligned[i] if vol_ma_aligned[i] > 0 else False
        
        if position == 0:
            # Long: weekly close below lower Bollinger Band with volume confirmation
            if close_1w[i] < lower_band_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: weekly close above upper Bollinger Band with volume confirmation
            elif close_1w[i] > upper_band_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: weekly close returns above weekly SMA (mean reversion)
            if close_1w[i] > sma.iloc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: weekly close returns below weekly SMA (mean reversion)
            if close_1w[i] < sma.iloc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals