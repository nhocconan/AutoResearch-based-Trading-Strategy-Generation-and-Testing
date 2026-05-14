#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Long when price breaks above H3 AND 1d close > 1d EMA34 (bullish regime)
- Short when price breaks below L3 AND 1d close < 1d EMA34 (bearish regime)
- Volume confirmation: current volume > 2.0 * 20-period average volume (strong spike)
- Exit on opposite Camarilla level (L3 for long exit, H3 for short exit)
- Uses 6h primary with 1d HTF to target 50-150 trades over 4 years (12-37/year)
- Camarilla levels provide mean-reversion structure; EMA34 filters regime; volume spike confirms momentum
- Designed to work in both bull (breakouts with trend) and bear (breakouts against trend) markets
- Signal size: 0.25 discrete levels to minimize fee churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels (based on previous period) on 6h data
    # H3 = close + 1.1 * (high - low) / 2
    # L3 = close - 1.1 * (high - low) / 2
    # Using previous bar's high/low/close for forward-looking calculation
    camarilla_h3 = np.roll(close, 1) + 1.1 * (np.roll(high, 1) - np.roll(low, 1)) / 2
    camarilla_l3 = np.roll(close, 1) - 1.1 * (np.roll(high, 1) - np.roll(low, 1)) / 2
    # Set first value to NaN since we don't have previous bar
    camarilla_h3[0] = np.nan
    camarilla_l3[0] = np.nan
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    ema_34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Trend filter: bullish if close > EMA34, bearish if close < EMA34
    bullish_regime = close > ema_34_1d_aligned
    bearish_regime = close < ema_34_1d_aligned
    
    # Volume confirmation: volume > 2.0 * 20-period average (strong spike)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(1, 34, 20)  # Need Camarilla (1), EMA34, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Camarilla H3 AND bullish regime AND volume confirmation
            if close[i] > camarilla_h3[i] and bullish_regime[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Camarilla L3 AND bearish regime AND volume confirmation
            elif close[i] < camarilla_l3[i] and bearish_regime[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below Camarilla L3 (opposite level)
            if close[i] < camarilla_l3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above Camarilla H3 (opposite level)
            if close[i] > camarilla_h3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0