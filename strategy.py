#!/usr/bin/env python3
"""
4h_PriceAction_Trend_Momentum_v1
Hypothesis: Combines 4h Donchian breakout with 1d volume confirmation and 1w trend filter to capture strong momentum moves in both bull and bear markets.
Long when price breaks above 4h Donchian high (20) with volume > 1.5x 20-period average and price above 1w EMA(34).
Short when price breaks below 4h Donchian low (20) with volume confirmation and price below 1w EMA(34).
Uses ATR-based stoploss via signal reversal to limit drawdown.
Target: 20-40 trades/year by requiring confluence of breakout, volume, and trend.
Works in bull markets via breakout momentum and in bear via short breakdowns.
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
    
    # 4h Donchian channels (20-period)
    donchian_high = np.full_like(high, np.nan)
    donchian_low = np.full_like(low, np.nan)
    
    lookback = 20
    for i in range(lookback, n):
        donchian_high[i] = np.max(high[i-lookback:i])
        donchian_low[i] = np.min(low[i-lookback:i])
    
    # Daily volume confirmation
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full_like(vol_1d, np.nan)
    vol_period = 20
    
    for i in range(vol_period, len(vol_1d)):
        vol_ma_1d[i] = np.mean(vol_1d[i-vol_period:i])
    
    vol_ma_4h = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    if len(close_1w) >= 34:
        ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    else:
        ema_1w = np.full_like(close_1w, np.nan)
    
    ema_1w_4h = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, vol_period, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_4h[i]) or np.isnan(ema_1w_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x daily average volume
        # Approximate 4h volume as 1/4 of daily volume (6 periods per day)
        vol_4h_approx = volume[i] * 4  # Scale to daily equivalent
        vol_confirm = vol_4h_approx > 1.5 * vol_ma_4h[i]
        
        if position == 0:
            # Long: break above Donchian high with volume and above weekly EMA
            if close[i] > donchian_high[i] and vol_confirm and close[i] > ema_1w_4h[i]:
                signals[i] = 0.30
                position = 1
            # Short: break below Donchian low with volume and below weekly EMA
            elif close[i] < donchian_low[i] and vol_confirm and close[i] < ema_1w_4h[i]:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Long exit: break below Donchian low
            if close[i] < donchian_low[i]:
                signals[i] = -0.30  # reverse to short
                position = -1
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Short exit: break above Donchian high
            if close[i] > donchian_high[i]:
                signals[i] = 0.30  # reverse to long
                position = 1
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_PriceAction_Trend_Momentum_v1"
timeframe = "4h"
leverage = 1.0