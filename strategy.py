#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band squeeze breakout with weekly trend filter and volume confirmation.
# Bollinger Band squeeze identifies low volatility periods that precede explosive moves.
# Uses weekly EMA50 for trend filter (avoid counter-trend trades) and volume spike for confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
# Works in both bull and bear markets by using weekly trend filter to align with higher timeframe trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # EMA(50) for weekly trend filter
    ema50_1w = np.zeros(len(close_1w))
    ema_multiplier = 2 / (50 + 1)
    ema50_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        ema50_1w[i] = (close_1w[i] - ema50_1w[i-1]) * ema_multiplier + ema50_1w[i-1]
    
    # Align weekly EMA to 12h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Bollinger Bands on 12h timeframe
    bb_period = 20
    bb_std = 2.0
    
    # Calculate Bollinger Bands using vectorized operations where possible
    sma = np.full(n, np.nan)
    std_dev = np.full(n, np.nan)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    bb_width = np.full(n, np.nan)
    
    for i in range(bb_period - 1, n):
        sma[i] = np.mean(close[i - bb_period + 1:i + 1])
        std_dev[i] = np.std(close[i - bb_period + 1:i + 1])
        upper_band[i] = sma[i] + bb_std * std_dev[i]
        lower_band[i] = sma[i] - bb_std * std_dev[i]
        bb_width[i] = (upper_band[i] - lower_band[i]) / sma[i] if sma[i] != 0 else 0
    
    # Bollinger Band squeeze detection: BB width below 20-period percentile
    bb_width_percentile = np.full(n, np.nan)
    lookback = 50
    for i in range(lookback, n):
        window = bb_width[i - lookback:i]
        if not np.all(np.isnan(window)):
            bb_width_percentile[i] = np.percentile(window[~np.isnan(window)], 20)
    
    # Volume confirmation: volume above 30-period average
    avg_volume = np.full(n, np.nan)
    vol_period = 30
    for i in range(vol_period, n):
        avg_volume[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(max(bb_period, lookback, vol_period) + 1, n):
        # Skip if any required data is not ready
        if (np.isnan(sma[i]) or np.isnan(std_dev[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(bb_width[i]) or 
            np.isnan(bb_width_percentile[i]) or np.isnan(avg_volume[i]) or
            np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        bb_w = bb_width[i]
        bb_w_percentile = bb_width_percentile[i]
        sma_val = sma[i]
        upper = upper_band[i]
        lower = lower_band[i]
        ema_trend = ema50_1w_aligned[i]
        
        # Bollinger Band squeeze condition: current width below 20th percentile of lookback period
        squeeze_condition = bb_w < bb_w_percentile
        
        # Breakout conditions
        breakout_up = price > upper
        breakout_down = price < lower
        
        # Volume confirmation: current volume > 2.0 x average volume
        volume_confirm = vol > 2.0 * avg_vol
        
        if position == 0:
            # Long: squeeze breakout upward + above weekly EMA50 + volume confirmation
            if squeeze_condition and breakout_up and price > ema_trend and volume_confirm:
                position = 1
                signals[i] = position_size
            # Short: squeeze breakout downward + below weekly EMA50 + volume confirmation
            elif squeeze_condition and breakout_down and price < ema_trend and volume_confirm:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle Bollinger Band (SMA) or squeeze ends
            if price <= sma_val or not squeeze_condition:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to middle Bollinger Band (SMA) or squeeze ends
            if price >= sma_val or not squeeze_condition:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w_BollingerSqueeze_Trend_Volume"
timeframe = "12h"
leverage = 1.0