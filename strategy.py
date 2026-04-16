#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d volume confirmation.
# Uses Bollinger Band width (20,2) on 6h to detect low volatility squeezes.
# Breakout occurs when price closes outside Bollinger Bands with volume > 1.5x average.
# Direction determined by 1d EMA(50) trend: long if price > EMA50, short if price < EMA50.
# Works in both bull (breakouts with trend) and bear (mean reversion in squeeze) markets.
# Target: 50-150 total trades over 4 years = 12-37/year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h data (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # === 1d data (higher timeframe for trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # === 6h Bollinger Bands (20,2) ===
    sma_20 = pd.Series(close_6h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_6h).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    bb_width = (bb_upper - bb_lower) / sma_20  # Normalized width
    
    # === 6h Bollinger Band squeeze detection ===
    # Squeeze when BB width is below 20-period low
    bb_width_low = pd.Series(bb_width).rolling(window=20, min_periods=20).min().values
    squeeze = bb_width <= bb_width_low
    
    # === 6h volume ratio for confirmation ===
    vol_ma_10_6h = pd.Series(volume_6h).rolling(window=10, min_periods=10).mean().values
    vol_ratio_6h = volume_6h / vol_ma_10_6h
    
    # === 1d EMA(50) for trend filter ===
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 6h timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_6h, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_6h, bb_lower)
    squeeze_aligned = align_htf_to_ltf(prices, df_6h, squeeze.astype(float))
    vol_ratio_aligned = align_htf_to_ltf(prices, df_6h, vol_ratio_6h)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(bb_upper_aligned[i]) or 
            np.isnan(bb_lower_aligned[i]) or
            np.isnan(squeeze_aligned[i]) or
            np.isnan(vol_ratio_aligned[i]) or
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        bb_upper = bb_upper_aligned[i]
        bb_lower = bb_lower_aligned[i]
        squeeze_val = squeeze_aligned[i]
        vol_ratio = vol_ratio_aligned[i]
        ema_50 = ema_50_aligned[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            # Stop loss: price closes below entry - 2.5 * ATR(14)
            # Approximate ATR using BB width
            atr_approx = (bb_upper - bb_lower) / 4  # Rough approximation
            if price < entry_price - 2.5 * atr_approx:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Stop loss: price closes above entry + 2.5 * ATR(14)
            atr_approx = (bb_upper - bb_lower) / 4
            if price > entry_price + 2.5 * atr_approx:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price touches opposite Bollinger Band or squeeze ends
            if price >= bb_lower or squeeze_val < 0.5:  # Touch lower band or squeeze ended
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price touches opposite Bollinger Band or squeeze ends
            if price <= bb_upper or squeeze_val < 0.5:  # Touch upper band or squeeze ended
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0 and squeeze_val > 0.5:  # Only enter during squeeze
            # Volume confirmation required
            if vol_ratio > 1.5:
                # Determine direction based on 1d EMA(50) trend
                if price > bb_upper and price > ema_50:  # Break above upper band in uptrend
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                elif price < bb_lower and price < ema_50:  # Break below lower band in downtrend
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_BB_Squeeze_Breakout_Volume_EMA50_v1"
timeframe = "6h"
leverage = 1.0