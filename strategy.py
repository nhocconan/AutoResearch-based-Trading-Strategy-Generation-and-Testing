#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above R3 AND close > 1d EMA50 (uptrend) AND volume > 1.8 * 20-bar avg volume
# Short when price breaks below S3 AND close < 1d EMA50 (downtrend) AND volume > 1.8 * 20-bar avg volume
# Exit when price reverts to the 1d VWAP (mean reversion to daily fair value)
# Uses discrete sizing 0.25 to control fee drag and drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Camarilla levels from 1d provide institutional support/resistance; EMA50 filters trend
# Volume spike confirms breakout authenticity; VWAP exit works in ranging and trending markets

name = "6h_Camarilla_R3S3_1dEMA50_Volume_VWAPExit_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla, EMA50, and VWAP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d VWAP (typical price * volume / cumulative volume)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    pv_1d = typical_price_1d * df_1d['volume'].values
    cum_pv_1d = np.cumsum(pv_1d)
    cum_vol_1d = np.cumsum(df_1d['volume'].values)
    vwap_1d = np.divide(cum_pv_1d, cum_vol_1d, out=np.zeros_like(cum_pv_1d), where=cum_vol_1d!=0)
    
    # Align HTF indicators to 6h timeframe (wait for completed HTF bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate volume confirmation: volume > 1.8 * 20-bar average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Camarilla breakout signals with trend and volume filters
            # Long: break above R3 AND uptrend AND volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 AND downtrend AND volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reverts to 1d VWAP (mean reversion)
            if close[i] <= vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reverts to 1d VWAP (mean reversion)
            if close[i] >= vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals