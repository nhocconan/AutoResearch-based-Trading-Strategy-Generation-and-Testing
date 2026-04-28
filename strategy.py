#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d ATR-based volume spike filter and trend filter
# Long when price breaks above R3 AND close > 1d EMA50 AND volume > 2.5x ATR(14)-scaled volume MA
# Short when price breaks below S3 AND close < 1d EMA50 AND volume > 2.5x ATR(14)-scaled volume MA
# Exit when price retouches opposite Camarilla level (S3 for longs, R3 for shorts)
# Uses ATR-scaled volume filter to adapt to volatility regimes, reducing false breakouts in low vol
# 12h timeframe targets 12-37 trades/year. Discrete sizing (0.30) minimizes fee churn.
# Works in bull via breakout+trend, works in bear via volume spike filter capturing panic climaxes

name = "12h_Camarilla_R3S3_Breakout_1dEMA50_ATR_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50, ATR, and Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate ATR(14) on 1d for volume scaling
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])  # First TR undefined, set to inf
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous 1d OHLC
    # R3 = close + ((high - low) * 1.1/4)
    # S3 = close - ((high - low) * 1.1/4)
    camarilla_r3 = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    camarilla_s3 = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    
    # Align 1d indicators to 12h timeframe (use completed 1d bar's values)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: >2.5x ATR-scaled 20-bar average volume (adaptive to volatility)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    # Scale volume MA by ATR to get volatility-adjusted threshold
    volume_threshold = 2.5 * volume_ma_20 * (atr_14_1d_aligned / np.maximum(close, 1e-10))
    volume_confirm = volume > volume_threshold
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_ma_20[i]) or
            np.isnan(atr_14_1d_aligned[i]) or atr_14_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_trend = ema_50_1d_aligned[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        curr_close = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above R3 AND close > 1d EMA50 AND volume confirmation
            if curr_close > r3_level and curr_close > ema_trend and vol_conf:
                signals[i] = 0.30
                position = 1
            # Short when price breaks below S3 AND close < 1d EMA50 AND volume confirmation
            elif curr_close < s3_level and curr_close < ema_trend and vol_conf:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price retouches S3 (opposite level)
            if curr_close <= s3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:  # Short - exit when price retouches R3 (opposite level)
            if curr_close >= r3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals