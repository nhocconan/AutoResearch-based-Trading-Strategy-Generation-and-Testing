#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_v2
Hypothesis: Daily Camarilla R3/S3 breakouts with weekly trend alignment (price vs 1w EMA50) and volume confirmation capture institutional participation. 
Choppiness Index filter avoids false breakouts in ranging markets. Discrete sizing (0.25) controls fee drawdown. Target: 50-100 trades over 4 years.
Works in bull via breakout momentum, in bear via mean-reversion touches of S3/R3 as rejection levels.
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
    
    # Get 1d data for Camarilla and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    range_1d = high_1d - low_1d
    
    # Prior day Camarilla R3/S3 levels (based on completed daily candle)
    camarilla_r3 = close_1d + 1.125 * range_1d
    camarilla_s3 = close_1d - 1.125 * range_1d
    
    # Weekly trend: 1w EMA50 on close
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: today's volume > 1.8 * 20-day average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_avg_20)
    
    # Choppiness Index (14) - avoid breakouts in choppy markets
    # True Range calculation
    tr1 = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr1 = np.maximum(tr1, np.abs(low - np.roll(close, 1)))
    tr1[0] = high[0] - low[0]
    tr_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    atr_14 = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values
    chop = 100 * np.log10(tr_sum / (atr_14 * 14)) / np.log10(14)
    chop_filter = chop < 61.8  # Only trade when not strongly ranging
    
    # Align HTF indicators to daily timeframe (price is already daily)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    chop_filter_aligned = align_htf_to_ltf(prices, df_1d, chop_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need Camarilla (1), EMA50 (50), volume avg (20), chop (14)
    start_idx = max(1, 50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_confirm_aligned[i]) or 
            np.isnan(chop_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        ema50 = ema50_1w_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        chop_ok = chop_filter_aligned[i]
        
        if position == 0:
            # Determine weekly trend alignment
            uptrend = close_val > ema50
            downtrend = close_val < ema50
            
            if uptrend and vol_conf and chop_ok:
                # Long bias: break above R3 with volume in uptrend
                if close_val > r3:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif downtrend and vol_conf and chop_ok:
                # Short bias: break below S3 with volume in downtrend
                if close_val < s3:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Long exit: stoploss (2.0*ATR) or touch of S3 (mean reversion)
            atr_approx = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values[i]
            stop_loss = entry_price - 2.0 * atr_approx
            
            if close_val <= stop_loss:
                signals[i] = 0.0
                position = 0
            elif close_val < s3:  # Reversion to S3
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: stoploss (2.0*ATR) or touch of R3 (mean reversion)
            atr_approx = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values[i]
            stop_loss = entry_price + 2.0 * atr_approx
            
            if close_val >= stop_loss:
                signals[i] = 0.0
                position = 0
            elif close_val > r3:  # Reversion to R3
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_v2"
timeframe = "1d"
leverage = 1.0