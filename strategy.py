#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA(34) trend filter and 1d volume spike confirmation.
- Primary timeframe: 1h for entry timing precision.
- HTF trend: 4h EMA(34) - bullish when price > EMA, bearish when price < EMA.
- HTF regime filter: 1d volume > 1.5 * 20-period volume MA to avoid low-volatility chop.
- Entry: Long when price breaks above Camarilla H3 level AND 4h EMA trend bullish AND 1d volume spike.
         Short when price breaks below Camarilla L3 level AND 4h EMA trend bearish AND 1d volume spike.
- Exit: Opposite Camarilla breakout (L3 for long, H3 for short) or loss of volume confirmation.
- Signal size: 0.20 discrete to limit drawdown and reduce fee churn.
- Session filter: 08-20 UTC to avoid Asian session noise.
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
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
    
    # Calculate Camarilla levels (based on previous bar's range)
    # H3 = close + 1.1 * (high - low) / 6
    # L3 = close - 1.1 * (high - low) / 6
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 6
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 6
    
    # Get 4h data for EMA(34) trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 4h close
    ema_34 = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_4h, ema_34)
    
    # Get 1d data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period volume MA on 1d
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 1h volume > 1.5 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 1)  # Need enough bars for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i]) or np.isnan(volume_spike[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_34_val = ema_34_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        h3_level = camarilla_h3[i]
        l3_level = camarilla_l3[i]
        vol_spike = volume_spike[i]
        in_sess = in_session[i]
        
        if position == 0:
            # Check for entry signals with volume spike and session filter
            if vol_spike and in_sess:
                # Bullish breakout: price breaks above H3 AND 4h EMA34 bullish (price > EMA)
                if curr_high > h3_level and curr_close > ema_34_val:
                    signals[i] = 0.20
                    position = 1
                # Bearish breakout: price breaks below L3 AND 4h EMA34 bearish (price < EMA)
                elif curr_low < l3_level and curr_close < ema_34_val:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: price breaks below L3 OR loss of volume confirmation OR outside session
            if curr_low < l3_level or not vol_spike or not in_sess:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above H3 OR loss of volume confirmation OR outside session
            if curr_high > h3_level or not vol_spike or not in_sess:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hEMA34Trend_1dVolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0