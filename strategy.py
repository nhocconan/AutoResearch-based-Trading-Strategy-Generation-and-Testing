#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeSpike
Hypothesis: Camarilla R3/S3 breakout on 6h timeframe with 1d EMA50 trend filter and volume spike confirmation (>2.0x average volume).
Trade only breakouts aligned with daily trend during strong volume expansion. Uses discrete sizing (0.25) and ATR-based stoploss.
Designed for 6h timeframe to achieve 12-37 trades/year (50-150 total over 4 years) while minimizing fee drag.
Works in both bull (trend-following breakouts) and bear (mean-reversion at extremes) regimes via daily EMA trend filter.
"""

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
    
    # Get 6h data for Camarilla levels and ATR - primary timeframe
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 2:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Camarilla levels for R3 and S3
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    camarilla_range = high_6h - low_6h
    r3_6h = close_6h + camarilla_range * 1.1 / 4
    s3_6h = close_6h - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_6h, r3_6h)
    s3_aligned = align_htf_to_ltf(prices, df_6h, s3_6h)
    
    # Get 1d data for EMA50 trend filter - HTF
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA to 6h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate ATR(14) for stoploss on 6h
    # True Range
    tr1 = high_6h[1:] - low_6h[1:]
    tr2 = np.abs(high_6h[1:] - close_6h[:-1])
    tr3 = np.abs(low_6h[1:] - close_6h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_6h = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_6h_aligned = align_htf_to_ltf(prices, df_6h, atr_6h)
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(50, 20, 14)  # EMA needs 50, vol needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or 
            np.isnan(atr_6h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_val = ema_1d_aligned[i]
        atr_val = atr_6h_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current volume > 2.0x 20-period average
        volume_spike = vol_val > 2.0 * vol_ma_val
        
        if position == 0:
            # Look for entry signals: Camarilla breakout with trend and volume confirmation
            # Long: price breaks above R3, above 1d EMA50, with volume spike
            long_signal = (high_val > r3_val) and (close_val > ema_val) and volume_spike
            # Short: price breaks below S3, below 1d EMA50, with volume spike
            short_signal = (low_val < s3_val) and (close_val < ema_val) and volume_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Stoploss: price moves against position by 2.0*ATR
            if close_val < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: price closes below 1d EMA50
            elif close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Stoploss: price moves against position by 2.0*ATR
            if close_val > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: price closes above 1d EMA50
            elif close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0