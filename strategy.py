#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Camarilla levels (H3, L3) act as intraday support/resistance derived from prior day's range
- Breakout above H3 or below L3 with volume > 2x average signals institutional participation
- 1d EMA34 ensures trades align with higher timeframe trend (avoid counter-trend in chop)
- Volume spike filter reduces false breakouts during low-liquidity periods
- Position size: 0.30 discrete level to balance return and drawdown
- Target: 25-50 trades/year on 4h timeframe (100-200 total over 4 years)
- Works in bull/bear via 1d trend filter and volatility-adjusted breakouts
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
    
    # Volume confirmation: > 2.0x 20-period average (tighter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Camarilla levels for today (based on prior 1d candle)
    # H3 = close + 1.1*(high - low)/4
    # L3 = close - 1.1*(high - low)/4
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 4.0
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 4.0
    
    # Align Camarilla levels to 4h timeframe (each level constant for the day)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where volume MA and 1d indicators are ready
    start_idx = max(20, 34)  # volume MA, 1d EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above H3 AND volume confirmation AND price above 1d EMA34 (uptrend)
            if close[i] > camarilla_h3_aligned[i] and volume_confirm and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below L3 AND volume confirmation AND price below 1d EMA34 (downtrend)
            elif close[i] < camarilla_l3_aligned[i] and volume_confirm and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price crosses back below H3 OR volume dries up OR trend reverses
            if close[i] < camarilla_h3_aligned[i] or not volume_confirm or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price crosses back above L3 OR volume dries up OR trend reverses
            if close[i] > camarilla_l3_aligned[i] or not volume_confirm or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0