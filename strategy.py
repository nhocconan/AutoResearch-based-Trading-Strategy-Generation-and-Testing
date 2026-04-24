#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 Breakout with 12h EMA34 Trend Filter and Volume Spike Confirmation.
- Camarilla pivot levels (H3/L3) act as magnet zones; breakouts beyond these levels indicate strong momentum.
- 12h EMA34 provides intermediate trend filter to align with higher-timeframe direction.
- Volume spike (>2.0x 20-period average) confirms breakout validity and reduces false signals.
- Position size 0.25 balances profit potential and drawdown control in volatile crypto markets.
- Target trades: 100-180 total over 4 years (25-45/year) to balance opportunity and fee drag.
- Works in bull/bear markets via 12h trend filter and volatility expansion logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA34 trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Camarilla pivot levels for 6h using previous bar's OHLC
    # H3 = Close + 1.1*(High-Low)/4, L3 = Close - 1.1*(High-Low)/4
    camarilla_h3 = close + 1.1 * (high - low) / 4
    camarilla_l3 = close - 1.1 * (high - low) / 4
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Only enter on volume spike
            if volume_spike:
                # Long: break above H3 + above 12h EMA34 (bullish intermediate trend)
                if close[i] > camarilla_h3[i] and close[i] > ema_34_12h_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: break below L3 + below 12h EMA34 (bearish intermediate trend)
                elif close[i] < camarilla_l3[i] and close[i] < ema_34_12h_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price closes below camarilla H3 level OR trend reverses
            if close[i] < camarilla_h3[i] or close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above camarilla L3 level OR trend reverses
            if close[i] > camarilla_l3[i] or close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_12hEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0