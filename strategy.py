#!/usr/bin/env python3
# Strategy: 1d_WeeklyEMA50_CamarillaBreakout_VolumeFilter
# Hypothesis: Daily breakout above weekly EMA50-defined support/resistance (weekly EMA50 ± ATR) with volume confirmation.
# Uses 1d bars for entries, filtered by weekly EMA50 trend to avoid counter-trend trades. Volume > 1.5x 20-day MA confirms institutional interest.
# Designed for 10-30 trades/year to minimize fee drag and work in both bull/bear markets.
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for EMA50 and ATR
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly EMA50 for trend
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Weekly ATR for dynamic bands
    high_low = high_1w - low_1w
    high_close = np.abs(high_1w - np.roll(close_1w, 1))
    low_close = np.abs(low_1w - np.roll(close_1w, 1))
    high_low[0] = high_1w[0] - low_1w[0]
    high_close[0] = np.abs(high_1w[0] - close_1w[0])
    low_close[0] = np.abs(low_1w[0] - close_1w[0])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_20_aligned = align_htf_to_ltf(prices, df_1w, atr_20)
    
    # Dynamic support/resistance: EMA50 ± 1.0 * ATR
    upper_band = ema50_1w_aligned + 1.0 * atr_20_aligned
    lower_band = ema50_1w_aligned - 1.0 * atr_20_aligned
    
    # Load daily data for entry timing, volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Volume spike detection (20-period on 1d)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        vol = volume_1d[i]
        
        if position == 0:
            # Long: price breaks above upper band with volume confirmation
            if (price > upper_band[i] and vol > 1.5 * vol_ma_20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band with volume confirmation
            elif (price < lower_band[i] and vol > 1.5 * vol_ma_20_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower band
            if price < lower_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above upper band
            if price > upper_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyEMA50_CamarillaBreakout_VolumeFilter"
timeframe = "1d"
leverage = 1.0