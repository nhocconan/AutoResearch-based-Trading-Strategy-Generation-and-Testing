#!/usr/bin/env python3
name = "6h_BollingerBands_WidthTrend_1dADX"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mfi_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h OHLC for trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h EMA50 trend
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    trend_up = close > ema_50_12h_aligned
    trend_down = close < ema_50_12h_aligned
    
    # Daily OHLC for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Bollinger Bands (20, 2)
    bb_ma_20 = pd.Series(daily_close).rolling(window=20, min_periods=20).mean().values
    bb_std_20 = pd.Series(daily_close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_ma_20 + (bb_std_20 * 2)
    bb_lower = bb_ma_20 - (bb_std_20 * 2)
    bb_width = bb_upper - bb_lower
    
    # Align Bollinger Bands width to 6h timeframe
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    
    # 1d ADX for trend strength
    tr = np.maximum(np.maximum(daily_high - daily_low, np.abs(daily_high - np.roll(daily_close, 1))), np.abs(daily_low - np.roll(daily_close, 1)))
    tr[0] = daily_high[0] - daily_low[0]
    dm_plus = np.where((daily_high - np.roll(daily_high, 1)) > (np.roll(daily_low, 1) - daily_low), np.maximum(daily_high - np.roll(daily_high, 1), 0), 0)
    dm_minus = np.where((np.roll(daily_low, 1) - daily_low) > (daily_high - np.roll(daily_high, 1)), np.maximum(np.roll(daily_low, 1) - daily_low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    di_plus = 100 * dm_plus14 / tr14
    di_minus = 100 * dm_minus14 / tr14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filter: current volume > 1.5x 20-period average (6h bars = 5 days)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 12  # ~3 days (12*6h) to prevent overtrading
    
    start_idx = 50  # BB and ADX need sufficient data
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bb_width_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine trend direction
        trending_up = trend_up[i]
        trending_down = trend_down[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Bollinger width contraction (low volatility) + ADX > 25 (trending) + price breaks above upper band in 12h uptrend
            if (bb_width_aligned[i] < np.percentile(bb_width_aligned[:i+1], 20) and  # Low volatility (bottom 20%)
                adx_aligned[i] > 25 and  # Strong trend
                close[i] > bb_upper_aligned[i] and 
                trending_up and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Bollinger width contraction + ADX > 25 + price breaks below lower band in 12h downtrend
            elif (bb_width_aligned[i] < np.percentile(bb_width_aligned[:i+1], 20) and 
                  adx_aligned[i] > 25 and 
                  close[i] < bb_lower_aligned[i] and 
                  trending_down and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls back below middle band or ADX drops below 20 (trend weakening)
            if close[i] < bb_ma_20_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises back above middle band or ADX drops below 20
            if close[i] > bb_ma_20_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: On 6h timeframe, Bollinger Bands width contraction (low volatility) combined with ADX > 25 (strong trend) and price breaking above/below Bollinger Bands with 12h EMA50 trend filter captures explosive moves after consolidation periods. This works in both bull and bear markets as it identifies volatility contractions that precede significant price expansions. The 12h trend filter ensures alignment with higher timeframe momentum, reducing false signals. Target: 50-150 trades over 4 years (12-37/year) to minimize fee decay while capturing high-probability breakouts. Volume confirmation ensures institutional participation.