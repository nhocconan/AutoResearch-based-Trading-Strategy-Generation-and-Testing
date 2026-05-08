# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using weekly RSI(14) for trend filter, daily Donchian(20) breakout, and volume confirmation.
# Long when weekly RSI > 50, price breaks above daily Donchian upper band, volume > 1.5x average.
# Short when weekly RSI < 50, price breaks below daily Donchian lower band, volume > 1.5x average.
# Uses ATR-based volatility sizing and time-based exits to limit drawdown.
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag.
# Works in bull (trend follow) and bear (trend still exists in downtrends).

name = "12h_weeklyRSI_dailyDonchian_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for RSI trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 14:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    
    # Get daily data for Donchian bands
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    
    # Weekly RSI(14)
    delta = np.diff(close_weekly, prepend=close_weekly[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    rsi_above_50 = rsi > 50
    
    # Daily Donchian(20) bands
    donchian_high = pd.Series(high_daily).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_daily).rolling(window=20, min_periods=20).min().values
    
    # Align weekly RSI to 12h
    rsi_above_50_aligned = align_htf_to_ltf(prices, df_weekly, rsi_above_50.astype(float))
    # Align daily Donchian bands to 12h
    donchian_high_aligned = align_htf_to_ltf(prices, df_daily, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_daily, donchian_low)
    
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
            # Long: weekly RSI > 50, price breaks above daily Donchian upper band, volume spike
            if (rsi_above_50_aligned[i] and
                close[i] > donchian_high_aligned[i] and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25 * vol_factor[i]
                position = 1
                entry_bar = i
            # Short: weekly RSI < 50, price breaks below daily Donchian lower band, volume spike
            elif (not rsi_above_50_aligned[i] and
                  close[i] < donchian_low_aligned[i] and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25 * vol_factor[i]
                position = -1
                entry_bar = i
        elif position == 1:
            # Long exit: RSI flip, price breaks below Donchian lower band, or max 30 bars held
            if (not rsi_above_50_aligned[i] or 
                close[i] < donchian_low_aligned[i] or
                i - entry_bar >= 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 * vol_factor[i]
        elif position == -1:
            # Short exit: RSI flip, price breaks above Donchian upper band, or max 30 bars held
            if (rsi_above_50_aligned[i] or 
                close[i] > donchian_high_aligned[i] or
                i - entry_bar >= 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25 * vol_factor[i]
    
    return signals