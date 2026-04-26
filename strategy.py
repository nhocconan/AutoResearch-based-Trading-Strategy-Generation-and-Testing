#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dEMA34_RegimeFilter_VolumeSpike_ATRStop_v2
Hypothesis: Camarilla R3/S3 breakout with 1d EMA34 trend filter, volume spike confirmation, choppiness regime filter, and ATR-based stoploss. Designed for low trade frequency (<50/year) to avoid fee drag while capturing strong trending moves in both bull and bear markets. Uses discrete position sizing (0.25) to minimize fee churn.
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
    
    # Get 1d data for Camarilla levels and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Camarilla R3 and S3 levels
    R3 = close_1d_prev + (high_1d - low_1d) * 1.1 / 4
    S3 = close_1d_prev - (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume confirmation: 2.0x average volume (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss (using 14-period ATR)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index regime filter (14-period)
    chop_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(chop_sum / (highest_high - lowest_low)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 1d EMA (34), volume MA (20), ATR (14), CHOP (14)
    start_idx = max(34, 20, 14, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_34_1d_val = ema_34_1d_aligned[i]
        R3_val = R3_aligned[i]
        S3_val = S3_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        chop_val = chop[i]
        
        # Regime filter: CHOP < 38.2 = trending (trend follow), CHOP > 61.8 = range (mean reversion)
        is_trending = chop_val < 38.2
        is_ranging = chop_val > 61.8
        
        if position == 0:
            # In trending regime: follow trend with breakout
            # In ranging regime: mean reversion at extremes
            if is_trending:
                # Long: price breaks above R3 with volume confirmation and uptrend
                long_signal = (high_val > R3_val) and (volume_val > 2.0 * vol_ma_val) and (close_val > ema_34_1d_val)
                # Short: price breaks below S3 with volume confirmation and downtrend
                short_signal = (low_val < S3_val) and (volume_val > 2.0 * vol_ma_val) and (close_val < ema_34_1d_val)
            else:  # ranging regime
                # Long: price rejects below S3 (mean reversion up) with volume confirmation
                long_signal = (low_val < S3_val) and (close_val > S3_val) and (volume_val > 2.0 * vol_ma_val)
                # Short: price rejects above R3 (mean reversion down) with volume confirmation
                short_signal = (high_val > R3_val) and (close_val < R3_val) and (volume_val > 2.0 * vol_ma_val)
            
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
            # Hold long
            signals[i] = 0.25
            # Exit: ATR stoploss or trend reversal or ranging regime exit signal
            if (close_val < entry_price - 2.5 * atr_val or 
                close_val < ema_34_1d_val or
                (is_ranging and close_val < (R3_val + S3_val) / 2)):  # exit at midpoint in ranging
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: ATR stoploss or trend reversal or ranging regime exit signal
            if (close_val > entry_price + 2.5 * atr_val or 
                close_val > ema_34_1d_val or
                (is_ranging and close_val > (R3_val + S3_val) / 2)):  # exit at midpoint in ranging
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_RegimeFilter_VolumeSpike_ATRStop_v2"
timeframe = "4h"
leverage = 1.0