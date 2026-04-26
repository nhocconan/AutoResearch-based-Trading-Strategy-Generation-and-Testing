#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_Regime_4hTrend
Hypothesis: 1h Camarilla R1/S1 breakout with regime filter (choppiness index) and 4h trend.
- Long when price breaks above Camarilla R1 AND choppy market (mean reversion) AND 4h EMA20 uptrend
- Short when price breaks below Camarilla S1 AND choppy market AND 4h EMA20 downtrend
- Uses prior 4h range for Camarilla levels (structure-based edge)
- Choppiness Index > 61.8 identifies ranging markets where mean reversion works
- 4h EMA20 filter ensures trading with higher timeframe trend
- Designed for low frequency (target 15-37 trades/year) with proven edge on BTC/ETH ranging markets
- Exit on opposite Camarilla level touch (S1 for longs, R1 for shorts)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for Camarilla levels and trend
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Camarilla levels from prior 4h bar
    # Camarilla: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    prior_4h_high = np.roll(df_4h['high'].values, 1)
    prior_4h_low = np.roll(df_4h['low'].values, 1)
    prior_4h_close = np.roll(df_4h['close'].values, 1)
    # First value is invalid due to roll
    prior_4h_high[0] = np.nan
    prior_4h_low[0] = np.nan
    prior_4h_close[0] = np.nan
    
    cam_r1 = prior_4h_close + (prior_4h_high - prior_4h_low) * 1.1 / 12
    cam_s1 = prior_4h_close - (prior_4h_high - prior_4h_low) * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe
    cam_r1_aligned = align_htf_to_ltf(prices, df_4h, cam_r1)
    cam_s1_aligned = align_htf_to_ltf(prices, df_4h, cam_s1)
    
    # Calculate 4h EMA20 for trend filter
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    # Trend: 1 = uptrend (close > EMA20), -1 = downtrend (close < EMA20), 0 = neutral
    trend_4h = np.where(ema_20_4h_aligned > 0, 
                        np.where(close > ema_20_4h_aligned, 1, -1), 
                        0)
    
    # Calculate Choppiness Index (14-period) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high,n) - min(low,n))))
    tr1 = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr2 = np.maximum(tr1, np.abs(low - np.roll(close, 1)))
    tr1[0] = np.nan  # First TR is undefined
    atr14 = pd.Series(tr2).rolling(window=14, min_periods=14).mean().values
    
    max_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range14 = max_high14 - min_low14
    
    # Avoid division by zero
    choppy = np.where(range14 > 0, 
                      100 * np.log10(atr14 * 14 / range14) / np.log10(14), 
                      100)  # Set to 100 when range is zero (max choppy)
    
    # Regime filter: CHOP > 61.8 = ranging market (favor mean reversion)
    ranging_market = choppy > 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for EMA, 14 for CHOP, 1 for prior 4h)
    start_idx = max(20, 14, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(cam_r1_aligned[i]) or np.isnan(cam_s1_aligned[i]) or
            np.isnan(ranging_market[i]) or np.isnan(trend_4h[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Session filter: 08-20 UTC (reduce noise trades)
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten or hold flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.0  # Exit long
                position = 0
            else:
                signals[i] = 0.0  # Exit short
                position = 0
            continue
        
        # Camarilla R1/S1 breakout conditions with regime filter and 4h trend filter
        if position == 0:
            # Long: Price breaks above Camarilla R1 AND ranging market AND 4h uptrend
            if close[i] > cam_r1_aligned[i] and ranging_market[i] and trend_4h[i] == 1:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below Camarilla S1 AND ranging market AND 4h downtrend
            elif close[i] < cam_s1_aligned[i] and ranging_market[i] and trend_4h[i] == -1:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: Price falls below Camarilla S1
            if close[i] < cam_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: Price rises above Camarilla R1
            if close[i] > cam_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_Regime_4hTrend"
timeframe = "1h"
leverage = 1.0