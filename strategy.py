#!/usr/bin/env python3
"""
1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeConfirm
Hypothesis: Camarilla R3/S3 breakouts on 1h aligned with 4h EMA50 trend and volume spikes capture high-probability moves in both bull and bear markets.
Uses 4h for signal direction (trend filter + volume confirmation) and 1h only for entry timing precision to minimize overtrading.
Session filter (08-20 UTC) reduces noise. Discrete sizing (0.20) controls fee drag and drawdown. Target: 60-150 total trades over 4 years.
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
    open_time = prices['open_time'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend and volume confirmation
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h volume average (20-period) for confirmation
    volume_4h = df_4h['volume'].values
    vol_avg_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_confirm_4h = volume_4h > (1.5 * vol_avg_4h)  # 4h volume > 1.5x 20-period average
    
    # Calculate 1h Camarilla levels (R3, S3) from prior 1h bar
    # Note: Camarilla typically uses daily, but we adapt to 1h using prior 1h bar's range
    high_1h = pd.Series(high).rolling(window=2, min_periods=2).max().shift(1).values  # prior 1h high
    low_1h = pd.Series(low).rolling(window=2, min_periods=2).min().shift(1).values   # prior 1h low
    close_1h = pd.Series(close).shift(1).values  # prior 1h close
    range_1h = high_1h - low_1h
    camarilla_r3 = close_1h + 1.125 * range_1h
    camarilla_s3 = close_1h - 1.125 * range_1h
    
    # Align all 4h indicators to 1h timeframe
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    volume_confirm_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_confirm_4h)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)  # Camarilla calculated from 1h but aligned via 4h structure? Wait, fix: Camarilla is 1h-based, no need to align via 4h
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Actually, Camarilla levels are 1h-indicators, so no HTF alignment needed for them
    # But we need to ensure we don't use look-ahead: they are based on prior bar, so safe
    # However, to be consistent with MTF approach and avoid look-ahead issues, we note:
    # The Camarilla levels are derived from 1h data and are for the current 1h bar based on prior bar -> valid at bar i
    
    # Re-align Camarilla properly: since they are 1h indicators based on prior bar, they are ready at bar i
    # No HTF alignment needed for 1h-based indicators
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.20   # Position size: 20% of capital (discrete level to minimize churn)
    
    # Warmup: need EMA50 (50), volume avg (20), Camarilla (prior bar needs 2 bars for high/low)
    start_idx = max(50, 20, 2)  # EMA50, volume avg, and we need at least 2 bars for prior high/low
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(volume_confirm_4h_aligned[i]) or 
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(close_1h[i]) or np.isnan(high_1h[i]) or np.isnan(low_1h[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3 = camarilla_r3[i]
        s3 = camarilla_s3[i]
        ema50 = ema50_4h_aligned[i]
        vol_conf = volume_confirm_4h_aligned[i]
        
        if position == 0:
            # Determine trend alignment: price vs 4h EMA50
            uptrend = close_val > ema50
            downtrend = close_val < ema50
            
            if uptrend and vol_conf:
                # Long bias: long when price breaks above R3 with volume confirmation
                if close_val > r3:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif downtrend and vol_conf:
                # Short bias: short when price breaks below S3 with volume confirmation
                if close_val < s3:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit conditions: stoploss (2.0*ATR) or Camarilla S3 touch
            atr_approx = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values[i]
            stop_loss = entry_price - 2.0 * atr_approx
            
            if close_val <= stop_loss:
                signals[i] = 0.0
                position = 0
            elif close_val < s3:  # Camarilla S3 touch
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit conditions: stoploss (2.0*ATR) or Camarilla R3 touch
            atr_approx = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values[i]
            stop_loss = entry_price + 2.0 * atr_approx
            
            if close_val >= stop_loss:
                signals[i] = 0.0
                position = 0
            elif close_val > r3:  # Camarilla R3 touch
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeConfirm"
timeframe = "1h"
leverage = 1.0