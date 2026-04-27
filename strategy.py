#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla R3/S3 breakout on 12h with 1d EMA34 trend filter and volume confirmation.
R3/S3 levels provide stronger breakout signals with lower frequency suitable for 12h timeframe.
1d trend filter avoids counter-trend whipsaws in both bull and bear markets.
Volume spike confirms breakout authenticity. Static exit on opposite Camarilla level touch.
Target: 50-150 trades over 4 years (12-37/year) with Sharpe > 1.0 on test.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR for Camarilla levels (20-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from previous 12h bar's OHLC
    prev_close = np.concatenate([[close[0]], close[:-1]])
    prev_high = np.concatenate([[high[0]], high[:-1]])
    prev_low = np.concatenate([[low[0]], low[:-1]])
    
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough for ATR, EMA34, volume average
    start_idx = max(100, 34, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        size = 0.25  # 25% position size
        
        if position == 0:
            # Flat - look for entry: breakout in direction of 1d trend with volume spike
            # Long: price breaks above Camarilla R3 AND 1d trend is up (close > EMA34) AND volume spike
            # Short: price breaks below Camarilla S3 AND 1d trend is down (close < EMA34) AND volume spike
            long_breakout = close_val > camarilla_r3[i]
            short_breakout = close_val < camarilla_s3[i]
            trend_up = close_val > ema_trend
            trend_down = close_val < ema_trend
            
            if long_breakout and trend_up and vol_spike:
                signals[i] = size
                position = 1
            elif short_breakout and trend_down and vol_spike:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Long - exit when price touches Camarilla S3 (failed breakout/reversal)
            if close_val < camarilla_s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price touches Camarilla R3 (failed breakout/reversal)
            if close_val > camarilla_r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0