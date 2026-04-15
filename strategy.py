#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band breakout with 12h Supertrend trend filter and volume confirmation
# Long when price breaks above upper BB(20,2) + 12h Supertrend = bullish + volume > 1.3x 20-period avg
# Short when price breaks below lower BB(20,2) + 12h Supertrend = bearish + volume > 1.3x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (20-40/year).
# Bollinger Bands adapt to volatility, providing dynamic breakout levels.
# Supertrend on 12h ensures we only trade with the higher timeframe trend, avoiding chop.
# Works in bull markets (riding uptrends) and bear markets (riding downtrends) by following 12h Supertrend direction.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # === 12h Indicator: Supertrend (ATR=10, mult=3.0) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ATR
    period = 10
    high_12h_shift = np.roll(high_12h, 1)
    low_12h_shift = np.roll(low_12h, 1)
    close_12h_shift = np.roll(close_12h, 1)
    high_12h_shift[0] = high_12h[0]
    low_12h_shift[0] = low_12h[0]
    close_12h_shift[0] = close_12h[0]
    
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - close_12h_shift)
    tr3 = np.abs(low_12h - close_12h_shift)
    tr1[0] = high_12h[0] - low_12h[0]
    tr2[0] = np.abs(high_12h[0] - close_12h[0])
    tr3[0] = np.abs(low_12h[0] - close_12h[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = np.zeros_like(tr)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    # Supertrend calculation
    hl2 = (high_12h + low_12h) / 2
    upperband = hl2 + (3.0 * atr)
    lowerband = hl2 - (3.0 * atr)
    
    supertrend = np.zeros_like(close_12h)
    direction = np.ones_like(close_12h)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upperband[0]
    direction[0] = 1
    
    for i in range(1, len(close_12h)):
        if close_12h[i] > supertrend[i-1]:
            direction[i] = 1
        else:
            direction[i] = -1
        
        if direction[i] == 1:
            supertrend[i] = max(lowerband[i], supertrend[i-1])
        else:
            supertrend[i] = min(upperband[i], supertrend[i-1])
    
    # 1 = uptrend, -1 = downtrend
    supertrend_direction = direction
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend_direction)
    
    # === 4h Indicator: Bollinger Bands (20, 2) ===
    bb_window = 20
    bb_std = 2.0
    sma = pd.Series(close).rolling(window=bb_window, min_periods=bb_window).mean().values
    std = pd.Series(close).rolling(window=bb_window, min_periods=bb_window).std().values
    upper_band = sma + (bb_std * std)
    lower_band = sma - (bb_std * std)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(bb_window, period) + 20  # BB(20) + Supertrend(10) + volume(20)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # Skip if any required data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(supertrend_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 4h Bollinger upper band (20,2)
        # 2. Trend (12h Supertrend = bullish, i.e., direction = 1)
        # 3. Volume confirmation
        if (close[i] > upper_band[i]) and \
           (supertrend_aligned[i] == 1) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 4h Bollinger lower band (20,2)
        # 2. Trend (12h Supertrend = bearish, i.e., direction = -1)
        # 3. Volume confirmation
        elif (close[i] < lower_band[i]) and \
             (supertrend_aligned[i] == -1) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_BB20_2_12hSupertrend10_3_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0