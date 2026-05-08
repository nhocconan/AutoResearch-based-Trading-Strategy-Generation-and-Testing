#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Keltner channel breakout with 1d trend filter and volume confirmation
# The Keltner channel (EMA + ATR) identifies volatility breakouts. Combined with
# daily EMA trend filter and volume confirmation, this strategy aims to capture
# strong momentum moves in both bull and bear markets. The ATR-based stop loss
# manages risk. Target: 20-30 trades/year per symbol.

name = "4h_KeltnerBreakout_1dTrend_VolumeConfirm"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 40-period EMA for Keltner center line
    ema40 = pd.Series(close).ewm(span=40, adjust=False, min_periods=40).mean().values
    
    # Calculate ATR(20) for Keltner channel width
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr20 = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Keltner channels (2.0 ATR multiplier)
    keltner_upper = ema40 + 2.0 * atr20
    keltner_lower = ema40 - 2.0 * atr20
    
    # Calculate volume confirmation (20-period average)
    vol_avg_20 = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:
            vol_avg_20[i] = np.mean(volume[i-19:i+1])
    
    # Align daily trend to 4h
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_20[i]) or np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_avg_today = vol_avg_20[i]
        vol_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirmed = vol_current > 1.5 * vol_avg_today
        
        if position == 0:
            # Long entry: price breaks above upper Keltner band with volume and trend confirmation
            if price > keltner_upper[i] and vol_confirmed and price > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower Keltner band with volume and trend confirmation
            elif price < keltner_lower[i] and vol_confirmed and price < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price closes below center line or trend changes
            if price < ema40[i] or price < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above center line or trend changes
            if price > ema40[i] or price > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals