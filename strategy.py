#!/usr/bin/env python3
# 12h_1d_1w_Camarilla_R3_S3_Breakout_Trend_Volume
# Hypothesis: Weekly (1w) and daily (1d) pivots define major support/resistance. On 12h timeframe,
# breakouts above R3 or below S3 are taken with 1d EMA34 trend filter and volume confirmation.
# Weekly trend (EMA50) filters direction: only long when above weekly EMA50, short when below.
# Volume surge (2x 20-period average) confirms breakout strength.
# Exit on opposite pivot touch or trend reversal.
# Target: 12-37 trades/year to minimize fee drag while capturing major moves in bull/bear markets.

name = "12h_1d_1w_Camarilla_R3_S3_Breakout_Trend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily and weekly data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 34 or len(df_1w) < 50:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily Camarilla pivot levels (R3, S3) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Calculate R3 and S3 levels
    r3 = pivot + 1.1 * range_1d / 2
    s3 = pivot - 1.1 * range_1d / 2
    
    # Shift by 1 to use only completed daily candle (avoid look-ahead)
    r3_prev = np.roll(r3, 1)
    s3_prev = np.roll(s3, 1)
    r3_prev[0] = np.nan
    s3_prev[0] = np.nan
    
    # Align daily R3/S3 to 12h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_prev)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_prev)
    
    # --- Daily EMA34 for trend filter ---
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # --- Weekly EMA50 for trend filter ---
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # --- Volume confirmation (2x 20-period average on 12h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for daily EMA34 (34), weekly EMA50 (50), and 20-period volume MA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume surge, daily uptrend, and above weekly EMA50
            if (close[i] > r3_aligned[i] and 
                volume_surge and 
                ema_34_aligned[i] < close[i] and
                ema_50_aligned[i] < close[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume surge, daily downtrend, and below weekly EMA50
            elif (close[i] < s3_aligned[i] and 
                  volume_surge and 
                  ema_34_aligned[i] > close[i] and
                  ema_50_aligned[i] > close[i]):
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price drops below S3 OR daily EMA34 turns down OR weekly EMA50 turns down
                if (close[i] < s3_aligned[i] or 
                    close[i] < ema_34_aligned[i] or
                    close[i] < ema_50_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises above R3 OR daily EMA34 turns up OR weekly EMA50 turns up
                if (close[i] > r3_aligned[i] or 
                    close[i] > ema_34_aligned[i] or
                    close[i] > ema_50_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals