#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Uses 4h EMA50 for structural trend bias (long when price > EMA50, short when price < EMA50)
# Camarilla R3/S3 levels provide precise entry/exit levels derived from 4h range
# Volume confirmation > 1.8x 20-period EMA ensures institutional participation
# Designed for low trade frequency: ~15-37 trades/year per symbol with 0.20 sizing
# 4h EMA50 filter reduces false breakouts in choppy markets while capturing strong trends
# Session filter (08-20 UTC) reduces noise trades outside active market hours
# Works in both bull and bear markets by following the dominant 4h trend

name = "1h_Camarilla_R3S3_4hEMA50_Trend_Volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h HTF data for EMA50 trend filter and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from previous 4h OHLC
    # Camarilla R3 = close + 1.1*(high-low)*1.1/4
    # Camarilla S3 = close - 1.1*(high-low)*1.1/4
    typical_range = df_4h['high'] - df_4h['low']
    camarilla_r3 = df_4h['close'] + 1.1 * typical_range * 1.1 / 4
    camarilla_s3 = df_4h['close'] - 1.1 * typical_range * 1.1 / 4
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3.values)
    
    # Volume confirmation: volume > 1.8 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need 4h data for EMA50 (50 bars) + Camarilla needs 4h
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 4h EMA50: long above EMA50, short below EMA50
        bullish_bias = close[i] > ema_50_4h_aligned[i]
        bearish_bias = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias:
                # Long: Camarilla R3 breakout above with volume spike
                if close[i] > camarilla_r3_aligned[i] and volume_spike[i]:
                    signals[i] = 0.20
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_bias:
                # Short: Camarilla S3 breakdown below with volume spike
                if close[i] < camarilla_s3_aligned[i] and volume_spike[i]:
                    signals[i] = -0.20
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop around EMA50
        
        elif position == 1:  # Long position
            # Exit: Camarilla S3 breakdown below (failure of breakout)
            if close[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Camarilla R3 breakout above (failure of breakdown)
            if close[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals