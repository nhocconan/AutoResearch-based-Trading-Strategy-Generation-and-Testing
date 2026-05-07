#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h RSI(14) with volume-weighted price change and 12h trend filter.
# Uses RSI for mean reversion in ranging markets and trend following in trending markets,
# filtered by 12h EMA to avoid counter-trend trades. Volume-weighted price change
# confirms momentum strength. Designed for low trade frequency (<30/year) to minimize
# fee drag while capturing both bull and bear market moves.
name = "4h_RSI_VolMom_12hTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h trend filter: 50-period EMA on close
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (alpha = 1/period)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume-weighted price change momentum
    price_change = (close - np.roll(close, 1)) / np.roll(close, 1)
    price_change[0] = 0
    vol_price_change = price_change * volume
    
    # Smooth volume-weighted price change
    vol_price_change_smooth = pd.Series(vol_price_change).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Momentum threshold (adaptive to volatility)
    vol_price_change_std = pd.Series(vol_price_change_smooth).rolling(50, min_periods=50).std().values
    vol_price_change_mean = pd.Series(vol_price_change_smooth).rolling(50, min_periods=50).mean().values
    mom_threshold = 0.5 * vol_price_change_std  # Half standard deviation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Sufficient warmup for RSI and moving averages
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_price_change_smooth[i]) or np.isnan(vol_price_change_std[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Momentum condition
        mom_long = vol_price_change_smooth[i] > mom_threshold[i]
        mom_short = vol_price_change_smooth[i] < -mom_threshold[i]
        
        if position == 0:
            # Long: RSI oversold with bullish momentum in uptrend
            long_condition = (rsi[i] < 30) and mom_long and uptrend
            # Short: RSI overbought with bearish momentum in downtrend
            short_condition = (rsi[i] > 70) and mom_short and downtrend
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI overbought or momentum turns bearish or trend fails
            if (rsi[i] > 70) or (not mom_long) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI oversold or momentum turns bullish or trend fails
            if (rsi[i] < 30) or (not mom_short) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals