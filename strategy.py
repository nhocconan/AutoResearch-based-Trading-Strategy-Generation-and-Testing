#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Daily data for Elder Ray and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Elder Ray components (21-period EMA)
    ema_21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    bull_power_1d = high_1d - ema_21_1d  # High - EMA21
    bear_power_1d = low_1d - ema_21_1d   # Low - EMA21
    
    # Align Elder Ray to 6h timeframe
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # 6h ATR(14) for volatility filter
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter - volume surge
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 1.5 * vol_ma20  # Moderate volume surge
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(21, n):  # Start after EMA warmup
        # Skip if data not ready
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull power positive AND bear power negative (bullish market structure)
            # With volume surge for confirmation
            if (bull_power_6h[i] > 0 and bear_power_6h[i] < 0 and vol_surge[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear power negative AND bull power negative (bearish market structure)
            # With volume surge for confirmation
            elif (bull_power_6h[i] < 0 and bear_power_6h[i] < 0 and vol_surge[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Market structure changes or volatility drops significantly
            if position == 1:
                # Exit long when bear power turns positive (bulls losing control)
                if bear_power_6h[i] >= 0 or atr[i] < 0.3 * atr[i]:  # Strong volatility drop
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short when bull power turns positive (bears losing control)
                if bull_power_6h[i] >= 0 or atr[i] < 0.3 * atr[i]:  # Strong volatility drop
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_MarketStructure_VolumeSurge_v1"
timeframe = "6h"
leverage = 1.0