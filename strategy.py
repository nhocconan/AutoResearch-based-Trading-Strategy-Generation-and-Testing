#!/usr/bin/env python3
"""
1d Camarilla Pivot Reversal with 1w EMA50 Trend Filter and Volume Spike
Hypothesis: Camarilla pivot levels (H3/L3) act as strong support/resistance on daily timeframe.
Price reversing from these levels with 1w EMA50 trend alignment and volume confirmation
captures mean-reversion moves in ranging markets and trend continuations in strong markets.
Designed for 1d timeframe to target 7-25 trades/year (30-100 over 4 years) by requiring
confluence of Camarilla H3/L3 touch, 1w EMA50 trend, and volume spike, reducing overtrading.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    ema_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate Camarilla pivot levels from previous day (using daily data)
    # We need to resample to daily to get proper OHLC for pivot calculation
    # But we cannot resample in loop - instead we calculate pivots using HTF daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day using previous day's OHLC
    # Camarilla: H3 = close + (high - low) * 1.1/4, L3 = close - (high - low) * 1.1/4
    # where close, high, low are from PREVIOUS day
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla H3 and L3
    camarilla_h3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_l3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels from daily to 1d timeframe (already aligned as we're on 1d)
    # Since we're on 1d timeframe, no alignment needed - values are already per bar
    camarilla_h3_aligned = camarilla_h3
    camarilla_l3_aligned = camarilla_l3
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA50 and volume MA
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1w EMA50
        bullish_bias = curr_close > ema_1w_aligned[i]
        bearish_bias = curr_close < ema_1w_aligned[i]
        
        if position == 0:
            # Look for entry signals - price touching Camarilla H3/L3 with volume spike
            # Long: price touches/below L3 AND bullish bias AND volume spike (mean reversion long)
            long_entry = (curr_low <= camarilla_l3_aligned[i]) and bullish_bias and vol_spike
            # Short: price touches/above H3 AND bearish bias AND volume spike (mean reversion short)
            short_entry = (curr_high >= camarilla_h3_aligned[i]) and bearish_bias and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price rises to midpoint (pivot) OR loss of bullish bias
            pivot = (camarilla_h3_aligned[i] + camarilla_l3_aligned[i]) / 2
            if (curr_close >= pivot) or (curr_close < ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price falls to midpoint (pivot) OR loss of bearish bias
            pivot = (camarilla_h3_aligned[i] + camarilla_l3_aligned[i]) / 2
            if (curr_close <= pivot) or (curr_close > ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_H3L3_Reversal_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0