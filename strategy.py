#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot (R1/S1) breakout with 1d volume spike and 12h EMA trend filter.
- Primary timeframe: 4h for execution.
- HTF: 1d for Camarilla levels and volume confirmation; 12h for EMA50 trend direction.
- Camarilla R1/S1 from previous 1d: strong intraday support/resistance levels.
- Entry: Long when price breaks above R1 AND 12h EMA50 up AND 1d volume spike (>1.5x 20-ma).
         Short when price breaks below S1 AND 12h EMA50 down AND 1d volume spike.
- Exit: Opposite Camarilla break (touch R1 for shorts, S1 for longs) or volume dries up.
- Volume confirmation avoids false breakouts in low-liquidity periods.
- Discrete signal size: 0.25 to balance return and drawdown.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Works in bull/bear: volume spike confirms institutional interest; EMA filter avoids counter-trend trades.
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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 12h data for EMA50 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day)
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    prev_close = df_1d['close'].shift(1)
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    camarilla_r1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    camarilla_s1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # Align 1d Camarilla to 4h (values from previous 1d bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1.values)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1.values)
    
    # 1d volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = df_1d['volume'].values > (1.5 * volume_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 50)  # Need enough 1d/12h bars
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(volume_spike_1d_aligned[i]) or np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        vol_spike = volume_spike_1d_aligned[i]
        ema_50 = ema_50_12h_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals with volume spike and EMA trend
            if vol_spike:
                # Bullish breakout: price breaks above R1 AND EMA50 rising (above prior)
                if curr_high > r1 and ema_50 > ema_50_12h_aligned[i-1]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below S1 AND EMA50 falling (below prior)
                elif curr_low < s1 and ema_50 < ema_50_12h_aligned[i-1]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price touches S1 (mean reversion) OR volume dries up
            if curr_low <= s1 or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price touches R1 (mean reversion) OR volume dries up
            if curr_high >= r1 or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_1dVolumeSpike_12hEMA50Trend_v1"
timeframe = "4h"
leverage = 1.0