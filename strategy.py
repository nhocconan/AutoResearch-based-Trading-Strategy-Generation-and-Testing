#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h/1d trend filter and volume confirmation.
- Long when price breaks above H3 AND 4h close > 4h EMA20 AND 1d close > 1d EMA34 (bullish regime)
- Short when price breaks below L3 AND 4h close < 4h EMA20 AND 1d close < 1d EMA34 (bearish regime)
- Volume confirmation: current volume > 2.0 * 20-period average volume (strong spike)
- Session filter: only trade between 08:00-20:00 UTC to avoid low-liquidity periods
- Exit on opposite Camarilla level (L3 for long exit, H3 for short exit)
- Uses 1h primary with 4h/1d HTF for direction, targeting 60-150 total trades over 4 years (15-37/year)
- Combines multiple timeframes for regime confirmation to reduce false breakouts
- Signal size: 0.20 discrete levels to minimize fee churn
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
    
    # Calculate Camarilla levels (based on previous period) on 1h data
    camarilla_h3 = np.roll(close, 1) + 1.1 * (np.roll(high, 1) - np.roll(low, 1)) / 2
    camarilla_l3 = np.roll(close, 1) - 1.1 * (np.roll(high, 1) - np.roll(low, 1)) / 2
    camarilla_h3[0] = np.nan
    camarilla_l3[0] = np.nan
    
    # Calculate 4h EMA20 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    htf_close = df_4h['close'].values
    ema_20_4h = pd.Series(htf_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    ema_34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Trend filters: bullish if close > EMA, bearish if close < EMA
    bullish_regime_4h = close > ema_20_4h_aligned
    bearish_regime_4h = close < ema_20_4h_aligned
    bullish_regime_1d = close > ema_34_1d_aligned
    bearish_regime_1d = close < ema_34_1d_aligned
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(1, 20, 34, 20)  # Camarilla (1), 4h EMA20, 1d EMA34, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above H3 AND bullish regime on both timeframes AND volume AND session
            if (close[i] > camarilla_h3[i] and bullish_regime_4h[i] and bullish_regime_1d[i] and 
                volume_confirm[i] and session_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: break below L3 AND bearish regime on both timeframes AND volume AND session
            elif (close[i] < camarilla_l3[i] and bearish_regime_4h[i] and bearish_regime_1d[i] and 
                  volume_confirm[i] and session_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: break below L3 (opposite level)
            if close[i] < camarilla_l3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: break above H3 (opposite level)
            if close[i] > camarilla_h3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hEMA20_1dEMA34_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0