#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend direction
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 for trend
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Get 1d data for ATR-based volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR uses only high-low
    
    # Calculate 14-day ATR
    atr_14 = np.full_like(tr, np.nan)
    for i in range(len(tr)):
        if i < 14:
            atr_14[i] = np.nan
        elif i == 14:
            atr_14[i] = np.mean(tr[1:15])
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Calculate ATR percentile rank over 60 days
    atr_percentile = np.full_like(atr_14, np.nan)
    for i in range(60, len(atr_14)):
        window = atr_14[i-60:i]
        valid_window = window[~np.isnan(window)]
        if len(valid_window) > 0:
            atr_percentile[i] = (np.sum(valid_window <= atr_14[i]) / len(valid_window)) * 100
    
    # Align ATR percentile to 1h timeframe
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # Volume confirmation: volume > 1.5x average volume (24-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=24, min_periods=24).mean().shift(1).values
    
    # Session filter: 8-20 UTC (pre-compute hours)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = max(60, 24, 50)  # 60 for ATR percentile, 24 for volume avg, 50 for EMA50
    
    for i in range(start, n):
        # Skip if any critical data is NaN or outside session
        if (np.isnan(atr_percentile_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(avg_vol[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        atr_percentile_val = atr_percentile_aligned[i]
        ema50_val = ema50_4h_aligned[i]
        
        if position == 0:
            # Low volatility regime: ATR percentile < 30 (bottom 30% of volatility)
            # Long: price above EMA50 with volume confirmation in low vol
            if price > ema50_val and vol > 1.5 * avg_vol[i] and atr_percentile_val < 30:
                position = 1
                signals[i] = position_size
            # Short: price below EMA50 with volume confirmation in low vol
            elif price < ema50_val and vol > 1.5 * avg_vol[i] and atr_percentile_val < 30:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below EMA50 or volatility expands (ATR percentile > 70)
            if price < ema50_val or atr_percentile_val > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above EMA50 or volatility expands (ATR percentile > 70)
            if price > ema50_val or atr_percentile_val > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_1d_VolatilityRegime_EMA50"
timeframe = "1h"
leverage = 1.0