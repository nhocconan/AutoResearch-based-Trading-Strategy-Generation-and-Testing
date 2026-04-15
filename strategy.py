#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot R3/S3 breakout with 1w EMA200 trend filter and volume confirmation
# Long when price breaks above Camarilla R3 + 1w EMA200 uptrend + volume > 2.0x 24-period avg
# Short when price breaks below Camarilla S3 + 1w EMA200 downtrend + volume > 2.0x 24-period avg
# Uses discrete position sizing (0.30) to control drawdown and minimize fee drag.
# 1w EMA200 provides strong long-term trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold (2.0x) targets ~15-30 trades/year on 12h timeframe to avoid overtrading.
# Camarilla pivots calculated from prior 12h bar's high/low/close for structure-based entries.

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
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # === 1w Indicator: EMA200 ===
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
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
    
    # Volume SMA for confirmation (using 24-period = 12d on 12h timeframe)
    vol_sma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(200, 24) + 5  # EMA200 + volume(24) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_sma_24[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 24-period volume SMA
        vol_confirm = volume[i] > (vol_sma_24[i] * 2.0)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R3 (close > R3)
        # 2. 1w EMA200 uptrend (close > EMA200)
        # 3. Volume confirmation
        if (close[i] > camarilla_r3[i]) and \
           (close[i] > ema_200_1w_aligned[i]) and vol_confirm:
            signals[i] = 0.30
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S3 (close < S3)
        # 2. 1w EMA200 downtrend (close < EMA200)
        # 3. Volume confirmation
        elif (close[i] < camarilla_s3[i]) and \
             (close[i] < ema_200_1w_aligned[i]) and vol_confirm:
            signals[i] = -0.30
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R3S3_1wEMA200_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0