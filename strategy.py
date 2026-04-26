#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_Chop_Filter
Hypothesis: On daily timeframe, Kaufman Adaptive Moving Average (KAMA) defines trend direction, RSI(14) filters overextended entries, and Choppiness Index (CHOP) regime filter ensures we only trade in trending markets (CHOP < 38.2) or ranging markets (CHOP > 61.8) with appropriate mean-reversion logic. Uses 1-week EMA50 as higher timeframe trend filter to avoid counter-trend trades. Designed for low trade frequency (7-25/year) to minimize fee drag and work in both bull and bear markets by adapting to regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data (primary timeframe is 1d, so prices themselves are 1d bars)
    # But we still need to load it via get_htf_data for consistency with MTF framework
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Load 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # KAMA (Kaufman Adaptive Moving Average) on 1d close
    # Efficiency ratio: ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=1) if len(close) > 1 else np.array([0])
    # For rolling sum of volatility, we need to compute properly
    volatility_rolling = pd.Series(np.abs(np.diff(close, n=1))).rolling(window=10, min_periods=10).sum().values
    # Pad the beginning with NaN for alignment
    change_padded = np.concatenate([np.full(9, np.nan), change])
    er = np.where(volatility_rolling != 0, change_padded / volatility_rolling, 0)
    # Smoothing constants: fastest SC = 2/(2+1) = 0.67, slowest SC = 2/(30+1) = 0.0645
    sc = (er * (0.67 - 0.0645) + 0.0645) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start after first 10 periods
    for i in range(10, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Align KAMA to 1d timeframe (already aligned since we computed on 1d)
    kama_aligned = kama  # No need to align as we used 1d data directly
    
    # 1w EMA50 for HTF trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # RSI(14) on 1d close
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad the beginning with NaN for first 14 periods
    rsi_padded = np.concatenate([np.full(14, np.nan), rsi])
    
    # Choppiness Index (CHOP) - 14 period
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    range_14 = highest_high_14 - lowest_low_14
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    chop = 100 * np.log10((atr_14 * 14) / range_14) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size (25% of capital)
    
    # Warmup: max of calculations (10 for KAMA, 14 for RSI/CHOP, 50 for 1w EMA)
    start_idx = max(10, 14, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(rsi_padded[i]) or 
            np.isnan(chop[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        kama_val = kama_aligned[i]
        ema_50_val = ema_50_1w_aligned[i]
        rsi_val = rsi_padded[i]
        chop_val = chop[i]
        
        # Determine 1d trend: bullish if price > KAMA, bearish if price < KAMA
        bullish_1d = close_val > kama_val
        bearish_1d = close_val < kama_val
        
        # Determine 1w trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_1w = close_val > ema_50_val
        bearish_1w = close_val < ema_50_val
        
        # Regime filters
        is_trending = chop_val < 38.2  # Trending regime
        is_ranging = chop_val > 61.8   # Ranging regime
        
        # Entry logic: adapt to regime
        if is_trending:
            # In trending regime: follow 1d trend with 1w filter and RSI not extreme
            long_entry = bullish_1d and bullish_1w and (rsi_val < 70)  # Not overbought
            short_entry = bearish_1d and bearish_1w and (rsi_val > 30)  # Not oversold
        elif is_ranging:
            # In ranging regime: mean reversion at extremes with RSI confirmation
            long_entry = (close_val < kama_val) and (rsi_val < 30)  # Oversold mean reversion long
            short_entry = (close_val > kama_val) and (rsi_val > 70)  # Overbought mean reversion short
        else:
            # Choppy middle: no trades
            long_entry = False
            short_entry = False
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = base_size
                position = 1
            elif short_entry:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit conditions
            if is_trending:
                # In trending regime: exit on trend change or RSI overbought
                if not bullish_1d or not bullish_1w or rsi_val > 80:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = base_size
            else:  # ranging regime
                # In ranging regime: exit on mean reversion to KAMA or RSI neutral
                if close_val > kama_val or rsi_val > 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = base_size
        elif position == -1:
            # Short - exit conditions
            if is_trending:
                # In trending regime: exit on trend change or RSI oversold
                if not bearish_1d or not bearish_1w or rsi_val < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -base_size
            else:  # ranging regime
                # In ranging regime: exit on mean reversion to KAMA or RSI neutral
                if close_val < kama_val or rsi_val < 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -base_size
    
    return signals

name = "1d_KAMA_Trend_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0