#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H4/L4 breakout with 1d volume confirmation and 1h RSI regime filter.
- Primary timeframe: 4h for execution, HTF: 1d for Camarilla pivots, 1h for RSI regime.
- RSI(14) > 60 indicates bullish momentum bias, RSI < 40 indicates bearish bias.
- Entry: Long when price breaks above H4 AND RSI > 50 (bullish breakout with momentum).
         Short when price breaks below L4 AND RSI < 50 (bearish breakout with momentum).
- Exit: Opposite Camarilla breakout (H4/L4) or RSI crosses 50 in opposite direction.
- Volume confirmation: current volume > 1.3 * 20-period volume MA (to avoid false breakouts).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-150 total trades over 4 years (19-37/year) for 4h timeframe.
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
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels (H4, L4) on 1d
    # Typical Price = (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    # Camarilla width = (H - L) * 1.1 / 8
    width = (df_1d['high'] - df_1d['low']) * 1.1 / 8.0
    # H4 = C + width * 1.1
    camarilla_h4 = df_1d['close'].values + width * 1.1
    # L4 = C - width * 1.1
    camarilla_l4 = df_1d['close'].values - width * 1.1
    
    # Get 1h data for RSI
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 30:
        return np.zeros(n)
    
    # Calculate RSI (14-period) on 1h
    delta = pd.Series(df_1h['close']).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align HTF indicators to 4h
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    rsi_aligned = align_htf_to_ltf(prices, df_1h, rsi)
    
    # Volume confirmation: current volume > 1.3 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # Need enough 1h bars for RSI and 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        rsi_val = rsi_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        h4 = camarilla_h4_aligned[i]
        l4 = camarilla_l4_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                # Bullish breakout: price closes above H4 AND RSI > 50 (bullish momentum)
                if curr_close > h4 and rsi_val > 50:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price closes below L4 AND RSI < 50 (bearish momentum)
                elif curr_close < l4 and rsi_val < 50:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price closes below L4 OR RSI drops below 50 (momentum loss)
            if curr_close < l4 or rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above H4 OR RSI rises above 50 (momentum shift)
            if curr_close > h4 or rsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H4L4_1dVolumeSpike_1hRSIRegime_v1"
timeframe = "4h"
leverage = 1.0