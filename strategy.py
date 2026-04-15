#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot R3/S3 breakout with 1d Williams %R extreme filter and volume confirmation
# Long when price breaks above Camarilla R3 + 1d Williams %R < -80 (oversold) + volume > 1.5x 24-period avg
# Short when price breaks below Camarilla S3 + 1d Williams %R > -20 (overbought) + volume > 1.5x 24-period avg
# Uses discrete position sizing (0.30) to balance risk and return.
# Williams %R on 1d provides mean-reversion edge in ranging markets while Camarilla breakouts capture momentum.
# Volume filter (1.5x) targets ~25-40 trades/year on 12h timeframe to avoid overtrading and fee drag.
# Camarilla R3/S3 levels provide stronger structure than R1/S1, reducing false breakouts.

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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # === 1d Indicator: Williams %R (14-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close_1d) / (highest_high - lowest_low)) * -100,
        -50.0  # neutral when range is zero
    )
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # === 12h Camarilla Pivot Levels (based on prior bar) ===
    # Pivot = (H + L + C) / 3
    # R3 = Pivot + (H - L) * 1.1 / 4
    # S3 = Pivot - (H - L) * 1.1 / 4
    # Using prior bar's OHLC to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    camarilla_r3 = pivot + (prev_high - prev_low) * 1.1 / 4.0
    camarilla_s3 = pivot - (prev_high - prev_low) * 1.1 / 4.0
    
    # Volume SMA for confirmation (using 24-period ~ 12h * 2 = 1 day)
    vol_sma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(24, 14) + 5  # volume(24) + Williams %R(14) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(williams_r_aligned[i]) or np.isnan(vol_sma_24[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 24-period volume SMA
        vol_confirm = volume[i] > (vol_sma_24[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R3 (close > R3)
        # 2. 1d Williams %R oversold (< -80)
        # 3. Volume confirmation
        if (close[i] > camarilla_r3[i]) and \
           (williams_r_aligned[i] < -80) and vol_confirm:
            signals[i] = 0.30
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S3 (close < S3)
        # 2. 1d Williams %R overbought (> -20)
        # 3. Volume confirmation
        elif (close[i] < camarilla_s3[i]) and \
             (williams_r_aligned[i] > -20) and vol_confirm:
            signals[i] = -0.30
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R3S3_1dWilliamsR_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0