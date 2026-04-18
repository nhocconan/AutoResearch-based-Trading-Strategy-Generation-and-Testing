#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 1d Keltner breakout with volume confirmation and ADX trend filter.
# Long when price breaks above upper Keltner (20, 2.0) with volume > 1.5x 20-period average and ADX > 20.
# Short when price breaks below lower Keltner with same conditions.
# Exit when price crosses back inside Keltner bands.
# Keltner channels adapt to volatility, reducing false breakouts in low volatility.
# ADX filter ensures trades only in trending markets, reducing whipsaw.
# Volume confirmation ensures institutional participation.
# Target: 20-40 trades/year for low fee drag and robust performance in bull/bear markets.
name = "4h_1dKeltner_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Keltner channels and ADX
    df_1d = get_htf_data(prices, '1d')
    
    # Typical price for Keltner: (H + L + C) / 3
    typical_price = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    
    # EMA of typical price (20-period)
    ema_tp = pd.Series(typical_price).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Average True Range (ATR) for Keltner width
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_1d)
    tr3 = np.abs(low_1d - close_1d)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner channels
    upper_keltner = ema_tp + (2.0 * atr)
    lower_keltner = ema_tp - (2.0 * atr)
    
    # ADX calculation (14-period)
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_di_14 = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / tr_14)
    minus_di_14 = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / tr_14)
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align all 1d indicators to 4h timeframe
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_keltner_aligned[i]) or np.isnan(lower_keltner_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper_keltner_val = upper_keltner_aligned[i]
        lower_keltner_val = lower_keltner_aligned[i]
        adx_val = adx_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: price breaks above upper Keltner with volume and trend
            if close_val > upper_keltner_val and vol_filter and adx_val > 20:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Keltner with volume and trend
            elif close_val < lower_keltner_val and vol_filter and adx_val > 20:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back inside Keltner bands
            if close_val < upper_keltner_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back inside Keltner bands
            if close_val > lower_keltner_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals