#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d trend filter and volume confirmation
# Long when price breaks above Camarilla R4 (1d) + 1d close > 1d EMA200 + volume > 1.3x avg
# Short when price breaks below Camarilla S4 (1d) + 1d close < 1d EMA200 + volume > 1.3x avg
# Uses discrete position sizing (0.25) to limit fee drag. Target: 15-30 trades/year.
# Works in bull/bear: breaks only with trend alignment avoids false breakouts in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # === 1d Indicators: Camarilla Pivots (based on previous day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar (using previous day's OHLC)
    camarilla_r4 = np.full(len(close_1d), np.nan)
    camarilla_s4 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        # Previous day's range
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_ = prev_high - prev_low
        
        if range_ > 0:
            camarilla_r4[i] = prev_close + range_ * 1.1 / 2
            camarilla_s4[i] = prev_close - range_ * 1.1 / 2
        else:
            camarilla_r4[i] = prev_close
            camarilla_s4[i] = prev_close
    
    # First bar has no previous day
    camarilla_r4[0] = close_1d[0]
    camarilla_s4[0] = close_1d[0]
    
    # === 1d Indicators: EMA200 for trend filter ===
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # === 1d Indicators: Volume SMA20 for confirmation ===
    volume_1d = df_1d['volume'].values
    vol_sma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 6h volume > 1.3x 1d average volume (scaled)
        # Approximate 1d volume to 6h by dividing by 4 (since 4x6h = 1d)
        vol_threshold = vol_sma20_1d_aligned[i] / 4.0 * 1.3
        vol_confirm = volume[i] > vol_threshold
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Camarilla R4
        # 2. 1d close > 1d EMA200 (uptrend)
        # 3. Volume confirmation
        if (close[i] > camarilla_r4_aligned[i]) and (close_1d[-1] > ema200_1d[-1]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Camarilla S4
        # 2. 1d close < 1d EMA200 (downtrend)
        # 3. Volume confirmation
        elif (close[i] < camarilla_s4_aligned[i]) and (close_1d[-1] < ema200_1d[-1]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Camarilla_R4_S4_Breakout_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0