#!/usr/bin/env python3

name = "6h_Choppiness_Regime_Adaptive"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for regime and trend filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 4h data for additional trend confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 1d Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]), np.abs(low_1d[1:] - close_1d[:-1]))
    tr = np.concatenate([[np.nan], tr1])
    
    # Sum of True Range over 14 periods
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(atr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero
    range_hl = hh - ll
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    chop = 100 * np.log10(atr_sum / range_hl) / np.log10(14)
    
    # 1d EMA50 for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 4h EMA20 for entry timing
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align indicators to 6h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Define regimes
    # Chop > 61.8 = ranging (mean revert)
    # Chop < 38.2 = trending (trend follow)
    chop_high = 61.8
    chop_low = 38.2
    
    ranging = chop_aligned > chop_high
    trending = chop_aligned < chop_low
    
    # Volume filter: current volume > 1.3x 20-period average (6h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 4  # ~1 day for 6h to reduce trades
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(chop_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(close_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine trend direction from 1d EMA50
        trend_up = close_1d_aligned[i] > ema_50_1d_aligned[i]
        trend_down = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # In ranging regime: mean reversion at Bollinger-like bands (using ATR)
            if ranging[i]:
                # Calculate 6-period ATR for band width
                tr6 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
                tr6_full = np.concatenate([[np.nan], tr6])
                atr6 = pd.Series(tr6_full).rolling(window=6, min_periods=6).mean().values
                atr6_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low, 'close': close}), atr6)
                
                if not np.isnan(atr6_aligned[i]) and atr6_aligned[i] > 0:
                    # Upper band: close + 1.5*ATR6
                    # Lower band: close - 1.5*ATR6
                    upper_band = close[i] + 1.5 * atr6_aligned[i]
                    lower_band = close[i] - 1.5 * atr6_aligned[i]
                    
                    # Long when price touches lower band in uptrend bias
                    if close[i] <= lower_band and trend_up and vol_filter[i]:
                        signals[i] = 0.25
                        position = 1
                        bars_since_last_trade = 0
                    # Short when price touches upper band in downtrend bias
                    elif close[i] >= upper_band and trend_down and vol_filter[i]:
                        signals[i] = -0.25
                        position = -1
                        bars_since_last_trade = 0
            
            # In trending regime: follow 4h EMA20 with pullback entry
            elif trending[i]:
                # Long when price pulls back to 4h EMA20 in uptrend
                if close[i] <= ema_20_4h_aligned[i] * 1.005 and trend_up and vol_filter[i]:
                    # Additional check: price above 4h EMA20 recently (pullback, not breakdown)
                    lookback = min(6, i)
                    if lookback > 0:
                        recent_close = close[i-lookback:i]
                        recent_ema = ema_20_4h_aligned[i-lookback:i]
                        if np.any(recent_close > recent_ema):  # Was above EMA recently
                            signals[i] = 0.25
                            position = 1
                            bars_since_last_trade = 0
                
                # Short when price pulls back to 4h EMA20 in downtrend
                elif close[i] >= ema_20_4h_aligned[i] * 0.995 and trend_down and vol_filter[i]:
                    # Additional check: price below 4h EMA20 recently (pullback, not breakout)
                    lookback = min(6, i)
                    if lookback > 0:
                        recent_close = close[i-lookback:i]
                        recent_ema = ema_20_4h_aligned[i-lookback:i]
                        if np.any(recent_close < recent_ema):  # Was below EMA recently
                            signals[i] = -0.25
                            position = -1
                            bars_since_last_trade = 0
        
        # Exit conditions
        elif position == 1:
            # Exit long: chop shifts to extreme ranging (overbought) or trend reversal
            if chop_aligned[i] > 70 or (chop_aligned[i] > chop_high and not trend_up):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: chop shifts to extreme ranging (oversold) or trend reversal
            if chop_aligned[i] > 70 or (chop_aligned[i] > chop_high and not trend_down):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Adaptive strategy using 1d Choppiness Index to detect market regime.
# In ranging markets (CHOP > 61.8): mean reversion at ATR-based bands with trend bias.
# In trending markets (CHOP < 38.2): pullback entries to 4h EMA20 with 1d trend filter.
# Uses volume confirmation and cooldown to limit trades. Designed to work in both
# bull and bear markets by adapting to regime. 6h timeframe balances signal quality
# and trade frequency (target: 15-35 trades/year). Chop > 70 triggers exit to avoid
# chop whipsaw. Avoids overtrading by using regime as primary filter.