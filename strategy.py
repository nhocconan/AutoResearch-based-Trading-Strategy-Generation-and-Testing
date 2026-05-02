#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation
# Uses 6h primary timeframe for balance of signal frequency and noise reduction
# Camarilla R3/S3 levels provide high-probability reversal/continuation zones from 12h calculation
# 12h EMA50 ensures alignment with higher timeframe trend to avoid counter-trend entries
# Volume spike (>2.0 * 24-period EMA) confirms institutional participation
# Designed for low trade frequency: ~15-25 trades/year per symbol with 0.25 sizing
# Works in bull markets via breakout continuation and bear markets via trend-following alignment
# Avoids overtrading by requiring confluence of price level, trend, and volume

name = "6h_Camarilla_R3S3_12hEMA50_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h HTF data for Camarilla levels and EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels (based on previous 12h bar)
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Previous 12h bar OHLC for Camarilla calculation
    prev_close_12h = np.roll(close_12h, 1)
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    prev_close_12h[0] = np.nan
    prev_high_12h[0] = np.nan
    prev_low_12h[0] = np.nan
    
    # Calculate Camarilla levels
    camarilla_r3 = np.full(len(close_12h), np.nan)
    camarilla_s3 = np.full(len(close_12h), np.nan)
    camarilla_r4 = np.full(len(close_12h), np.nan)
    camarilla_s4 = np.full(len(close_12h), np.nan)
    
    for i in range(1, len(close_12h)):
        hl_range = prev_high_12h[i] - prev_low_12h[i]
        camarilla_r3[i] = prev_close_12h[i] + (hl_range * 1.1 / 4)
        camarilla_s3[i] = prev_close_12h[i] - (hl_range * 1.1 / 4)
        camarilla_r4[i] = prev_close_12h[i] + (hl_range * 1.1 / 2)
        camarilla_s4[i] = prev_close_12h[i] - (hl_range * 1.1 / 2)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h indicators to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 2.0 * 24-period EMA (6h * 4 = 24 periods = 4 days)
    vol_series = pd.Series(volume)
    vol_ema_24 = vol_series.ewm(span=24, adjust=False, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ema_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient data for all indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 12h EMA50
        bullish_bias = close[i] > ema_50_12h_aligned[i]
        bearish_bias = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias:
                # Long: price breaks above Camarilla R3 with volume spike
                if close[i] > camarilla_r3_aligned[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_bias:
                # Short: price breaks below Camarilla S3 with volume spike
                if close[i] < camarilla_s3_aligned[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop around EMA50
        
        elif position == 1:  # Long position
            # Exit: price breaks below Camarilla S3 or price below 12h EMA50
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Camarilla R3 or price above 12h EMA50
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals