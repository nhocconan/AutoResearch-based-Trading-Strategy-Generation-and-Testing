#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with volume spike and 4h trend filter
# Long when price breaks above 1h Camarilla R3 + volume > 2.0x 20-period avg + 4h close > 4h EMA34
# Short when price breaks below 1h Camarilla S3 + volume > 2.0x 20-period avg + 4h close < 4h EMA34
# Uses Camarilla pivots from 1h OHLC (no look-ahead) and 4h EMA for trend direction
# Designed for low trade frequency (15-35/year) to minimize fee drag while capturing institutional breakouts
# Session filter: 08-20 UTC to avoid low-volume Asian session noise

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 40:
        return np.zeros(n)
    
    # === 4h Indicators: EMA34 for trend filter ===
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    signals = np.zeros(n)
    
    # Precompute session hours (08-20 UTC) for efficiency
    hours = prices.index.hour
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Session filter: only trade during 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # === 1h Camarilla Pivot Levels (R3, S3) ===
        # Based on previous 1h bar's OHLC (no look-ahead)
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        pivot = (prev_high + prev_low + prev_close) / 3.0
        r3 = pivot + (prev_high - prev_low) * 1.1 / 2.0
        s3 = pivot - (prev_high - prev_low) * 1.1 / 2.0
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1h Camarilla R3
        # 2. Volume spike confirmation
        # 3. 4h trend filter: close > EMA34 (uptrend)
        if (close[i] > r3) and vol_confirm and (close[i] > ema_34_4h_aligned[i]):
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1h Camarilla S3
        # 2. Volume spike confirmation
        # 3. 4h trend filter: close < EMA34 (downtrend)
        elif (close[i] < s3) and vol_confirm and (close[i] < ema_34_4h_aligned[i]):
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_Camarilla_R3S3_VolumeSpike_4hEMA34_v1"
timeframe = "1h"
leverage = 1.0