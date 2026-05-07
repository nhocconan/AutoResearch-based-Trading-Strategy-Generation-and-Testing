#!/usr/bin/env python3
name = "6h_TRIX_Trend_Filtered_Momentum"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # TRIX (15-period) - momentum oscillator
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) - then % change
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix_raw = ema3.pct_change() * 100  # Convert to percentage
    trix = trix_raw.fillna(0).values
    
    # Align TRIX to 6h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # Daily trend filter: EMA(50) on daily close
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 20-period average (20 * 6h = 5 days)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(15*3, 20)  # TRIX needs 3*15=45 periods, volume MA 20
    
    for i in range(start_idx, n):
        if (np.isnan(trix_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX positive and rising, price above daily EMA50, volume confirmation
            trix_rising = trix_aligned[i] > trix_aligned[i-1]
            price_above_ema = close[i] > ema_50_1d_aligned[i]
            vol_condition = volume[i] > vol_ma_20[i] * 1.5
            
            if trix_rising and price_above_ema and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: TRIX negative and falling, price below daily EMA50, volume confirmation
            elif trix_aligned[i] < 0 and trix_aligned[i] < trix_aligned[i-1] and close[i] < ema_50_1d_aligned[i] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TRIX turns negative or volume drops
            if trix_aligned[i] < 0 or volume[i] < vol_ma_20[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX turns positive or volume drops
            if trix_aligned[i] > 0 or volume[i] < vol_ma_20[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6s TRIX momentum with daily trend filter and volume confirmation
# - TRIX (15,15,15) filters out market noise and identifies sustained momentum
# - Long when TRIX is rising above zero with price above daily EMA50 and volume confirmation
# - Short when TRIX is falling below zero with price below daily EMA50 and volume confirmation
# - Daily EMA50 provides trend filter to avoid counter-trend trades
# - Volume confirmation (1.5x average) ensures institutional participation
# - Exit when TRIX crosses zero or volume drops below 1.2x average
# - Position size 0.25 targets 15-30 trades/year, avoiding fee drag
# - Works in bull markets (catch momentum) and bear markets (catch declines)