#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w RSI(14) as trend filter, 1d Donchian(20) breakout, and volume confirmation.
# Long when 1w RSI > 50, price breaks above 1d Donchian upper band, volume > 1.5x average.
# Short when 1w RSI < 50, price breaks below 1d Donchian lower band, volume > 1.5x average.
# Includes volatility-based position sizing and time-based exits to limit drawdown.
# Target: 30-100 total trades over 4 years (7-25/year) to balance opportunity and fee drag.
# Works in bull (trend follow) and bear (trend still exists in downtrends).

name = "1d_1wRSI_1dDonchian_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for RSI trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Get 1d data for Donchian bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1w RSI(14)
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    rsi_above_50 = rsi > 50
    
    # 1d Donchian(20) bands
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1w RSI to 1d
    rsi_above_50_aligned = align_htf_to_ltf(prices, df_1w, rsi_above_50.astype(float))
    # Align 1d Donchian bands to 1d
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    # Volatility-based position sizing (ATR-based)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    vol_factor = np.clip(atr / (close * 0.01), 0.5, 2.0)  # Normalize volatility
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    
    start_idx = 34  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_above_50_aligned[i]) or np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(vol_factor[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 1w RSI > 50, price breaks above 1d Donchian upper band, volume spike
            if (rsi_above_50_aligned[i] and
                close[i] > donchian_high_aligned[i] and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25 * vol_factor[i]
                position = 1
                entry_bar = i
            # Short: 1w RSI < 50, price breaks below 1d Donchian lower band, volume spike
            elif (not rsi_above_50_aligned[i] and
                  close[i] < donchian_low_aligned[i] and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25 * vol_factor[i]
                position = -1
                entry_bar = i
        elif position == 1:
            # Long exit: RSI flip, price breaks below Donchian lower band, or max 30 days held
            if (not rsi_above_50_aligned[i] or 
                close[i] < donchian_low_aligned[i] or
                i - entry_bar >= 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 * vol_factor[i]
        elif position == -1:
            # Short exit: RSI flip, price breaks above Donchian upper band, or max 30 days held
            if (rsi_above_50_aligned[i] or 
                close[i] > donchian_high_aligned[i] or
                i - entry_bar >= 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25 * vol_factor[i]
    
    return signals