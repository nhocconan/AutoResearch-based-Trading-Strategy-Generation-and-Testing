#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H4/L4 breakout with 4h EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h EMA50 for trend filter (price > EMA50 = uptrend, price < EMA50 = downtrend).
- Entry: Long when price breaks above Camarilla H4 AND price > 4h EMA50 AND volume > 2.0 * 1h volume MA(20);
         Short when price breaks below Camarilla L4 AND price < 4h EMA50 AND volume > 2.0 * 1h volume MA(20).
- Exit: Opposite Camarilla breakout (Long exits when price < Camarilla L3, Short exits when price > Camarilla H3).
- Signal size: 0.20 discrete to balance capture and fee control.
- Uses H4/L4 (stronger levels) for fewer, higher-quality breakouts; EMA50 filters higher-timeframe trend; volume spike confirms conviction.
- Session filter: 08-20 UTC to reduce noise trades.
- Works in bull (buying strong breakouts) and bear (selling strong breakdowns) with reduced whipsaws from 4h trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Get 4h data for Camarilla pivot levels (based on previous 4h OHLC)
    if len(df_4h) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 4h OHLC
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ranges
    range_4h = high_4h - low_4h
    
    # Calculate Camarilla levels
    camarilla_h4 = close_4h + 1.5 * range_4h
    camarilla_l4 = close_4h - 1.5 * range_4h
    camarilla_h3 = close_4h + 1.25 * range_4h
    camarilla_l3 = close_4h - 1.25 * range_4h
    
    # Align Camarilla levels to 1h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    # Calculate 1h volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (precompute hours array)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 1)  # EMA50 needs 50 periods
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter: price > EMA50 = uptrend, price < EMA50 = downtrend
        uptrend = curr_close > ema_50_aligned[i]
        downtrend = curr_close < ema_50_aligned[i]
        
        # Volume confirmation: 2.0x threshold
        vol_confirm = curr_volume > 2.0 * vol_ma[i]
        
        if position == 0:
            # Check for entry signals
            if uptrend and vol_confirm:
                # Long: price breaks above Camarilla H4
                if curr_high > camarilla_h4_aligned[i]:
                    signals[i] = 0.20
                    position = 1
            elif downtrend and vol_confirm:
                # Short: price breaks below Camarilla L4
                if curr_low < camarilla_l4_aligned[i]:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long position: exit when price breaks below Camarilla L3
            if curr_low < camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position: exit when price breaks above Camarilla H3
            if curr_high > camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H4L4_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0