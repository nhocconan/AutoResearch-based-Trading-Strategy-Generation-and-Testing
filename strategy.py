#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Trade 4h Camarilla R3/S3 breakouts with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above R3 in bull regime (close > 1d EMA34) with volume spike.
Short when price breaks below S3 in bear regime (close < 1d EMA34) with volume spike.
Use ATR-based stoploss and discrete sizing (0.25) to limit trades to ~20-40/year.
Camarilla levels provide institutional support/resistance; 1d trend ensures directional alignment;
volume spike confirms institutional participation. Works in bull (longs) and bear (shorts).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend regime
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR for volatility normalization and volume spike
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 4h Camarilla levels from previous day's OHLC
    # Need to align daily OHLC to 4h bars
    df_1d_ohlc = df_1d[['open', 'high', 'low', 'close']]
    # Align each OHLC series to 4h timeframe
    open_1d = align_htf_to_ltf(prices, df_1d, df_1d_ohlc['open'].values)
    high_1d = align_htf_to_ltf(prices, df_1d, df_1d_ohlc['high'].values)
    low_1d = align_htf_to_ltf(prices, df_1d, df_1d_ohlc['low'].values)
    close_1d = align_htf_to_ltf(prices, df_1d, df_1d_ohlc['close'].values)
    
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    # S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    range_1d = high_1d - low_1d
    camarilla_r3 = close_1d + 1.1 * range_1d
    camarilla_s3 = close_1d - 1.1 * range_1d
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    # Start index: need warmup for 1d EMA34 (34) and ATR (14)
    start_idx = max(34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i]) or
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        # Volume spike: current volume > 2.0 * ATR (adaptive threshold)
        volume_spike = volume[i] > 2.0 * atr[i]
        
        # Determine 1d trend regime
        # Bull regime: close > EMA34
        # Bear regime: close < EMA34
        if close[i] > ema_34_1d_aligned[i]:
            regime = 'bull'
        elif close[i] < ema_34_1d_aligned[i]:
            regime = 'bear'
        else:
            regime = 'range'
        
        if position == 0:
            # Long setup: price breaks above R3 AND volume spike AND bull regime
            long_setup = (close[i] > camarilla_r3[i]) and volume_spike and (regime == 'bull')
            
            # Short setup: price breaks below S3 AND volume spike AND bear regime
            short_setup = (close[i] < camarilla_s3[i]) and volume_spike and (regime == 'bear')
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            elif short_setup:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
                bars_since_entry = 0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            bars_since_entry += 1
            # Exit: price closes below R3 OR regime turns bearish OR max holding period (12 bars = 2 days)
            if (close[i] < camarilla_r3[i]) or (regime == 'bear') or (bars_since_entry >= 12):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            bars_since_entry += 1
            # Exit: price closes above S3 OR regime turns bullish OR max holding period (12 bars = 2 days)
            if (close[i] > camarilla_s3[i]) or (regime == 'bull') or (bars_since_entry >= 12):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0