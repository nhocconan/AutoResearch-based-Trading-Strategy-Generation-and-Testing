#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 1w EMA34 trend filter and 1d volume spike confirmation.
- Primary timeframe: 6h for entries/exits.
- HTF: 1w EMA34 for trend direction (bullish if price > EMA34, bearish if price < EMA34).
- Volume: Current 6h volume > 2.0 * 20-period 1d volume MA to avoid false breakouts.
- Entry: Long when price breaks above Camarilla H3 AND 1w EMA34 trend bullish AND volume spike.
         Short when price breaks below Camarilla L3 AND 1w EMA34 trend bearish AND volume spike.
- Exit: Opposite Camarilla breakout (L3 for long, H3 for short) or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Why should work in bull/bear: Camarilla levels adapt to volatility, EMA34 filter avoids counter-trend trades, volume confirmation reduces false breakouts.
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
    
    # Calculate Camarilla levels (H3, L3) on 6h using previous bar's range
    # Camarilla: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    # Use previous bar to avoid look-ahead
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan  # first bar has no previous
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    camarilla_high = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_low = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Get 1w data for EMA34 trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1w close
    ema_34 = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 1d data for volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period volume MA on 1d
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 6h volume > 2.0 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 1)  # Need enough 1w bars for EMA34, 1d for volume MA, and 1 for Camarilla
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_high[i]) or 
            np.isnan(camarilla_low[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_34_val = ema_34_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        upper_camarilla = camarilla_high[i]
        lower_camarilla = camarilla_low[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish breakout: price breaks above upper Camarilla AND 1w EMA34 bullish (price > EMA34)
                if curr_high > upper_camarilla and ema_34_val > 0 and curr_close > ema_34_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below lower Camarilla AND 1w EMA34 bearish (price < EMA34)
                elif curr_low < lower_camarilla and ema_34_val > 0 and curr_close < ema_34_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below lower Camarilla OR loss of volume confirmation
            if curr_low < lower_camarilla or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above upper Camarilla OR loss of volume confirmation
            if curr_high > upper_camarilla or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_1wEMA34Trend_1dVolumeSpike_v1"
timeframe = "6h"
leverage = 1.0