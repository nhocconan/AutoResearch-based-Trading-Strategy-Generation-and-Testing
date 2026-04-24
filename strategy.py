#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d Supertrend trend filter and volume spike confirmation.
- Long when price breaks above Camarilla H3 AND 1d Supertrend is bullish AND volume > 2.0 * 20-period average
- Short when price breaks below Camarilla L3 AND 1d Supertrend is bearish AND volume > 2.0 * 20-period average
- Exit on opposite Camarilla level (exit long on L3, exit short on H3)
- Supertrend uses ATR(10) with multiplier 3.0 for robust trend detection
- Uses 4h primary with 1d HTF to target 75-200 total trades over 4 years (19-50/year)
- Supertrend adapts to volatility better than EMA, reducing whipsaws in ranging markets
- Signal size: 0.25 discrete levels to minimize fee churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla H3 and L3 levels using previous day's OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels: H3 = close + (high - low) * 1.1/4, L3 = close - (high - low) * 1.1/4
    camarilla_h3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_l3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align daily Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 1d Supertrend for trend filter
    hl2 = (df_1d['high'] + df_1d['low']) / 2
    atr = pd.Series(abs(df_1d['high'] - df_1d['low'])).rolling(window=10, min_periods=10).mean()
    upperband = hl2 + (3.0 * atr)
    lowerband = hl2 - (3.0 * atr)
    
    supertrend = np.zeros(len(df_1d))
    direction = np.ones(len(df_1d))  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(df_1d)):
        if close.iloc[i] > upperband.iloc[i-1]:
            direction[i] = 1
        elif close.iloc[i] < lowerband.iloc[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lowerband.iloc[i] < lowerband.iloc[i-1]:
                lowerband.iloc[i] = lowerband.iloc[i-1]
            if direction[i] == -1 and upperband.iloc[i] > upperband.iloc[i-1]:
                upperband.iloc[i] = upperband.iloc[i-1]
    
        if direction[i] == 1:
            supertrend[i] = lowerband.iloc[i]
        else:
            supertrend[i] = upperband.iloc[i]
    
    # Align Supertrend to 4h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    
    # Trend filter: bullish if close > Supertrend, bearish if close < Supertrend
    bullish_regime = close > supertrend_aligned
    bearish_regime = close < supertrend_aligned
    
    # Volume confirmation: volume > 2.0 * 20-period average (strong spike)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20) + 1  # Need EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(supertrend_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Camarilla H3 AND bullish regime AND volume confirmation
            if close[i] > camarilla_h3_aligned[i] and bullish_regime[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Camarilla L3 AND bearish regime AND volume confirmation
            elif close[i] < camarilla_l3_aligned[i] and bearish_regime[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below Camarilla L3 (opposite level)
            if close[i] < camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above Camarilla H3 (opposite level)
            if close[i] > camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_1dSupertrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0