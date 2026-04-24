#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 12h EMA34 trend filter and volume confirmation.
- Long when price breaks above H3 level AND 12h close > 12h EMA34 (bullish regime) AND volume > 1.5 * 20-period average
- Short when price breaks below L3 level AND 12h close < 12h EMA34 (bearish regime) AND volume > 1.5 * 20-period average
- Exit on opposite Camarilla level (L3 for long exit, H3 for short exit)
- Uses 4h primary with 12h HTF to target 75-200 trades over 4 years (19-50/year)
- Camarilla levels provide precise intraday support/resistance; EMA34 filters regime; volume confirms breakout strength
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
    
    # Calculate Camarilla levels (based on previous period's range)
    # H3 = close + 1.1 * (high - low) / 2
    # L3 = close - 1.1 * (high - low) / 2
    # Using previous bar's high/low/close to avoid look-ahead
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # first bar uses current values
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Calculate 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    daily_close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(daily_close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Trend filter: bullish if close > EMA34, bearish if close < EMA34
    bullish_regime = close > ema_34_12h_aligned
    bearish_regime = close < ema_34_12h_aligned
    
    # Volume confirmation: volume > 1.5 * 20-period average (moderate spike)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34)  # Need volume MA(20) and EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above H3 level AND bullish regime AND volume confirmation
            if close[i] > camarilla_h3[i] and bullish_regime[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below L3 level AND bearish regime AND volume confirmation
            elif close[i] < camarilla_l3[i] and bearish_regime[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below L3 level (opposite level)
            if close[i] < camarilla_l3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above H3 level (opposite level)
            if close[i] > camarilla_h3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_12hEMA34_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0