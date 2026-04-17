#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla pivot R1/S1 breakout with 1d EMA50 trend filter and volume confirmation
- Uses 1d EMA50 as HTF trend filter (smooth, reliable for regime identification)
- Camarilla R1/S1 levels provide precise intraday support/resistance from prior day
- Breakout above R1 in uptrend or below S1 in downtrend captures momentum with validation
- Volume spike (2.0x 20-period MA) confirms institutional participation
- Designed for 6b timeframe to balance trade frequency and capture meaningful moves
- Works in bull markets (buying R1 breakouts in uptrend) and bear markets (selling S1 breakdowns in downtrend)
- Target: 50-120 total trades over 4 years (~12-30/year) to minimize fee drag
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
    
    # Get 1d data for EMA50 trend filter and Camarilla pivot calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 6h data for primary timeframe (volume, price)
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate EMA50 on 1d for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla pivot levels from prior 1d candle
    # Camarilla: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    camarilla_range = (high_1d - low_1d) * 1.1 / 12
    camarilla_r1 = close_1d + camarilla_range
    camarilla_s1 = close_1d - camarilla_range
    
    # Volume average (20-period) on 6h
    volume_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema_trend = ema50_1d_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and trend alignment
            # Long: price breaks above Camarilla R1 + volume spike + price > 1d EMA50 (uptrend)
            if price > r1 and vol > 2.0 * vol_ma and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 + volume spike + price < 1d EMA50 (downtrend)
            elif price < s1 and vol > 2.0 * vol_ma and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price retracement below Camarilla S1 (mean reversion) or trend reversal
            if price < s1 or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price retracement above Camarilla R1 (mean reversion) or trend reversal
            if price > r1 or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R1S1_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0