#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: 12h Camarilla R3/S3 breakout with daily EMA34 trend filter and volume spike confirmation.
- Long when price breaks above Camarilla R3 AND daily EMA34 uptrend AND volume > 1.8 * volume_ma(30)
- Short when price breaks below Camarilla S3 AND daily EMA34 downtrend AND volume > 1.8 * volume_ma(30)
- Camarilla levels derived from 1d chart for key intraday support/resistance
- Daily EMA34 filter ensures trading with higher timeframe trend to avoid counter-trend whipsaws
- Volume spike (1.8x) confirms participation and reduces false breakouts
- Designed for low frequency (target 12-37 trades/year on 12h) to minimize fee drag
- Exit on opposite Camarilla level (S3 for longs, R3 for shorts) or trend reversal
- Novelty: Uses 12h timeframe with Camarilla from 1d, volume confirmation, and trend filter - avoids saturated 4h/6h Camarilla families
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla levels and trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from 1d chart (needs completed daily candle)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # But we only need R3 and S3: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    camarilla_r3 = daily_close + 1.1 * (daily_high - daily_low)
    camarilla_s3 = daily_close - 1.1 * (daily_high - daily_low)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate daily EMA34 for trend filter (needs completed daily candle)
    ema_34_1d = pd.Series(daily_close).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    # Trend: 1 = uptrend (close > EMA34), -1 = downtrend (close < EMA34), 0 = neutral/invalid
    trend_1d = np.where(ema_34_1d_aligned > 0, 
                        np.where(close > ema_34_1d_aligned, 1, -1), 
                        0)
    
    # Calculate volume filter: volume > 1.8 * volume_ma(30) for confirmation
    volume_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for daily EMA, 30 for volume MA)
    start_idx = max(34, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(trend_1d[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla breakout conditions with trend and volume spike filter
        if position == 0:
            # Long: Price breaks above Camarilla R3 AND daily uptrend AND volume spike
            if close[i] > camarilla_r3_aligned[i] and trend_1d[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3 AND daily downtrend AND volume spike
            elif close[i] < camarilla_s3_aligned[i] and trend_1d[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Camarilla S3 OR daily trend turns down
            if close[i] < camarilla_s3_aligned[i] or trend_1d[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Camarilla R3 OR daily trend turns up
            if close[i] > camarilla_r3_aligned[i] or trend_1d[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0