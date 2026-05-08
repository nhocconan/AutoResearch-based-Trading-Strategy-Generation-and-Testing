#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d MACD trend filter, 4h Donchian(20) breakout, and volume confirmation.
# Long when 1d MACD line > signal line, price breaks above 4h Donchian upper band, volume > 1.8x average.
# Short when 1d MACD line < signal line, price breaks below 4h Donchian lower band, volume > 1.8x average.
# Includes ATR-based position sizing (0.25-0.35) and max 20-bar hold to limit drawdown.
# Works in bull (trend follow) and bear (trend still exists in downtrends) due to MACD trend filter.
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag.

name = "4h_1dMACD_4hDonchian_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for MACD trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Get 4h data for Donchian bands
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 1d MACD(12,26,9)
    ema12 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(close_1d).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_above_signal = macd_line > signal_line
    
    # 4h Donchian(20) bands
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 1d MACD signal to 4h
    macd_above_signal_aligned = align_htf_to_ltf(prices, df_1d, macd_above_signal.astype(float))
    # Align 4h Donchian bands to 4h
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    # ATR(14) for position sizing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    # Normalize ATR to price for sizing factor (0.5 to 1.5)
    vol_factor = np.clip(atr / (close * 0.02), 0.5, 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(macd_above_signal_aligned[i]) or np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(vol_factor[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 1d MACD bullish, price breaks above 4h Donchian upper band, volume spike
            if (macd_above_signal_aligned[i] and
                close[i] > donchian_high_aligned[i] and
                vol_ratio[i] > 1.8):
                signals[i] = 0.25 * vol_factor[i]
                position = 1
                entry_bar = i
            # Short: 1d MACD bearish, price breaks below 4h Donchian lower band, volume spike
            elif (not macd_above_signal_aligned[i] and
                  close[i] < donchian_low_aligned[i] and
                  vol_ratio[i] > 1.8):
                signals[i] = -0.25 * vol_factor[i]
                position = -1
                entry_bar = i
        elif position == 1:
            # Long exit: MACD flip bearish, price breaks below Donchian lower band, or max 20 bars held
            if (not macd_above_signal_aligned[i] or 
                close[i] < donchian_low_aligned[i] or
                i - entry_bar >= 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 * vol_factor[i]
        elif position == -1:
            # Short exit: MACD flip bullish, price breaks above Donchian upper band, or max 20 bars held
            if (macd_above_signal_aligned[i] or 
                close[i] > donchian_high_aligned[i] or
                i - entry_bar >= 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25 * vol_factor[i]
    
    return signals