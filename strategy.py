#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1w EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 12h for entries/exits.
- HTF: 1w EMA34 for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Volume: Current 12h volume > 2.0 * 20-period volume MA to avoid false breakouts.
- Entry: Long when price breaks above Camarilla H3 level AND 1w EMA34 trend bullish AND volume spike.
         Short when price breaks below Camarilla L3 level AND 1w EMA34 trend bearish AND volume spike.
- Exit: Opposite Camarilla breakout or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Why: Camarilla levels provide intraday support/resistance, EMA34 filters major trend, volume spike confirms momentum.
       Works in bull (breakouts with trend) and bear (fades at extremes with volume exhaustion).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels (H3, L3) from previous 12h bar
    # H3 = close + 1.1*(high - low)/4
    # L3 = close - 1.1*(high - low)/4
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan  # First bar has no previous
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 4
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 4
    
    # Get 1w data for EMA34 trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1w close
    ema_34 = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume MA on 1w
    vol_ma_1w = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # Volume confirmation: current 12h volume > 2.0 * 20-period 1w volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_1w_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 1)  # Need enough 1w bars for EMA34 and volume MA (plus 1 for previous bar)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_34_val = ema_34_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        h3_level = camarilla_h3[i]
        l3_level = camarilla_l3[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish breakout: price breaks above H3 AND 1w EMA34 bullish (close > EMA34)
                if curr_high > h3_level and curr_close > ema_34_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below L3 AND 1w EMA34 bearish (close < EMA34)
                elif curr_low < l3_level and curr_close < ema_34_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below L3 OR loss of volume confirmation
            if curr_low < l3_level or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above H3 OR loss of volume confirmation
            if curr_high > h3_level or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_1wEMA34Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0