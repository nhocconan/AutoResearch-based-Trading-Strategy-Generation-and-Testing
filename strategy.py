#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4-hour EMA trend filter with volume spike and Camarilla pivot breakout
# Long when price breaks above 4h Camarilla R3 + volume > 1.8x 20-bar volume SMA + 4h EMA34 > EMA89 (bullish trend)
# Short when price breaks below 4h Camarilla S3 + volume > 1.8x 20-bar volume SMA + 4h EMA34 < EMA89 (bearish trend)
# Uses 1h for entry timing, 4h for trend and pivot structure. Discrete position size 0.20 to minimize fee churn.
# Designed for low trade frequency (15-35/year) with session filter (08-20 UTC) to avoid noise.
# Works in bull markets via trend continuation and bear markets via strong breakdowns with volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # === 4h Indicator: EMA34 and EMA89 for trend filter ===
    close_4h = df_4h['close'].values
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_4h = pd.Series(close_4h).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    ema89_4h_aligned = align_htf_to_ltf(prices, df_4h, ema89_4h)
    
    # === 4h Indicator: Camarilla Pivot Levels (R3, S3) from daily OHLC ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_for_pivot = df_4h['close'].values  # Use 4h close for pivot calculation (standard practice)
    
    camarilla_r3_4h = close_4h_for_pivot + (high_4h - low_4h) * 1.1 / 4
    camarilla_s3_4h = close_4h_for_pivot - (high_4h - low_4h) * 1.1 / 4
    
    camarilla_r3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    camarilla_s3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    
    # === 1h Indicator: Volume SMA for entry timing precision ===
    vol_sma_20_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20_1h[i] * 1.8)
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_4h_aligned[i]) or np.isnan(camarilla_s3_4h_aligned[i]) or
            np.isnan(ema34_4h_aligned[i]) or np.isnan(ema89_4h_aligned[i]) or
            np.isnan(vol_sma_20_1h[i])):
            signals[i] = 0.0
            continue
        
        # Determine 4h trend: bullish if EMA34 > EMA89, bearish if EMA34 < EMA89
        bullish_trend = ema34_4h_aligned[i] > ema89_4h_aligned[i]
        bearish_trend = ema34_4h_aligned[i] < ema89_4h_aligned[i]
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 4h Camarilla R3 level
        # 2. Bullish 4h trend (EMA34 > EMA89)
        # 3. Volume confirmation
        if (close[i] > camarilla_r3_4h_aligned[i]) and \
           bullish_trend and vol_confirm:
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 4h Camarilla S3 level
        # 2. Bearish 4h trend (EMA34 < EMA89)
        # 3. Volume confirmation
        elif (close[i] < camarilla_s3_4h_aligned[i]) and \
             bearish_trend and vol_confirm:
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_4hCamarillaR3S3_Volume_EMA34_89_Trend_v1"
timeframe = "1h"
leverage = 1.0