#!/usr/bin/env python3
"""
1d_1w_12h_Camarilla_R3_S3_Breakout_Trend_Volume
Hypothesis: Uses weekly trend filter (price above/below weekly 12-period EMA) combined with daily Camarilla R3/S3 level breakouts on 1d timeframe. Entry requires volume confirmation (1.5x 24-period volume average). Designed for low trade frequency (7-25/year) to avoid fee drag. Works in bull/bear markets by following weekly trend direction.
"""

name = "1d_1w_12h_Camarilla_R3_S3_Breakout_Trend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    # Typical price
    typical = (high + low + close) / 3
    # Camarilla levels
    R4 = close + ((high - low) * 1.5000)
    R3 = close + ((high - low) * 1.2500)
    R2 = close + ((high - low) * 1.1666)
    R1 = close + ((high - low) * 1.0833)
    PP = typical
    S1 = close - ((high - low) * 1.0833)
    S2 = close - ((high - low) * 1.1666)
    S3 = close - ((high - low) * 1.2500)
    S4 = close - ((high - low) * 1.5000)
    return R3, S3  # We only need R3 and S3 for breakout

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 1d OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly Trend Filter (12-period EMA) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 12:
        return np.zeros(n)
    
    ema_12_1w = pd.Series(df_1w['close'].values).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_12_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_12_1w)
    
    # --- Daily Camarilla R3/S3 Levels ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    r3_1d, s3_1d = calculate_camarilla(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # --- Volume Confirmation (24-period average) ---
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = np.divide(volume, vol_ma, out=np.ones_like(volume), where=vol_ma!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_12_1w_aligned[i]) or 
            np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Trend condition: price above/below weekly EMA12
        above_weekly_ema = close[i] > ema_12_1w_aligned[i]
        below_weekly_ema = close[i] < ema_12_1w_aligned[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: price above weekly EMA + breaks above R3 + volume
            if (above_weekly_ema and 
                close[i] > r3_1d_aligned[i] and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly EMA + breaks below S3 + volume
            elif (below_weekly_ema and 
                  close[i] < s3_1d_aligned[i] and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Camarilla level or trend reversal
            if position == 1:
                # Exit long: price breaks below S3 OR trend turns bearish
                if close[i] < s3_1d_aligned[i] or not above_weekly_ema:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above R3 OR trend turns bullish
                if close[i] > r3_1d_aligned[i] or not below_weekly_ema:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals