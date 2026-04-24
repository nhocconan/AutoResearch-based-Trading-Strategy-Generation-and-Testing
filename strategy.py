#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 Breakout with 12h EMA50 Trend Filter and Volume Spike Confirmation.
- Camarilla pivot levels (R3/S3) act as strong intraday support/resistance.
- Breakout beyond R3/S3 + volume confirmation captures institutional moves.
- 12h EMA50 provides higher-timeframe trend filter to avoid counter-trend trades.
- Position size 0.25 balances profit and drawdown control (max 77% BTC drop → ~19% equity loss).
- Target trades: 100-200 total over 4 years (25-50/year) to balance opportunity and fee drag.
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
    
    # Get 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels from previous 1d candle
    # R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), S3 = C - ((H-L)*1.1/4)
    camarilla_R3 = np.full(n, np.nan)
    camarilla_S3 = np.full(n, np.nan)
    
    for i in range(1, n):
        prev_high = df_1d['high'].iloc[i-1] if i-1 < len(df_1d) else df_1d['high'].iloc[-1]
        prev_low = df_1d['low'].iloc[i-1] if i-1 < len(df_1d) else df_1d['low'].iloc[-1]
        prev_close = df_1d['close'].iloc[i-1] if i-1 < len(df_1d) else df_1d['close'].iloc[-1]
        
        camarilla_R3[i] = prev_close + ((prev_high - prev_low) * 1.1 / 4)
        camarilla_S3[i] = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    
    # Volume confirmation: > 2.0x 20-period average (tighter for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_R3[i]) or 
            np.isnan(camarilla_S3[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Only trade with volume confirmation
            if volume_confirm:
                # Long: break above R3 + above 12h EMA50 (bullish higher-timeframe trend)
                if close[i] > camarilla_R3[i] and close[i] > ema_50_12h_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: break below S3 + below 12h EMA50 (bearish higher-timeframe trend)
                elif close[i] < camarilla_S3[i] and close[i] < ema_50_12h_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price closes below S3 (reversion to mean) OR trend change
            if close[i] < camarilla_S3[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above R3 (reversion to mean) OR trend change
            if close[i] > camarilla_R3[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0