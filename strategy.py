#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 1d volume spike and 1w EMA50 trend filter.
- Primary timeframe: 6h for execution, HTF: 1d for Camarilla levels and volume confirmation, HTF: 1w for EMA50 trend.
- Camarilla levels calculated from prior 1d OHLC: H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4.
- Entry: Long when price breaks above H3 with volume spike (>1.5x 20-period MA) AND 1w EMA50 up (close > EMA50).
         Short when price breaks below L3 with volume spike AND 1w EMA50 down (close < EMA50).
- Exit: Opposite Camarilla break (H3/L3) or loss of volume confirmation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels (H3, L3) from prior 1d OHLC
    # H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    prior_close = df_1d['close'].shift(1).values
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    camarilla_h3 = prior_close + 1.1 * (prior_high - prior_low) / 4.0
    camarilla_l3 = prior_close - 1.1 * (prior_high - prior_low) / 4.0
    
    # Align 1d Camarilla to 6h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 1d)
    volume_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = df_1d['volume'].values > (1.5 * volume_ma_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w close
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 50)  # Need enough 1d and 1w bars
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        vol_spike = volume_spike_aligned[i]
        ema50 = ema_50_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_spike:
                # Bullish breakout: price breaks above H3 AND 1w EMA50 up
                if curr_high > h3 and curr_close > ema50:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below L3 AND 1w EMA50 down
                elif curr_low < l3 and curr_close < ema50:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below L3 OR loss of volume spike
            if curr_low < l3 or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above H3 OR loss of volume spike
            if curr_high > h3 or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_CamarillaH3L3_1dVolumeSpike_1wEMA50Trend_v1"
timeframe = "6h"
leverage = 1.0