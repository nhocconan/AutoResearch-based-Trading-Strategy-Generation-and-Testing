#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA34 trend filter and volume spike.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h EMA(34) for trend filter (defines bull/bear regime).
- Entry: Long when price breaks above Camarilla H3 in bull regime with volume > 2.0 * 1h volume MA(20);
         Short when price breaks below Camarilla L3 in bear regime with volume > 2.0 * 1h volume MA(20).
- Exit: Opposite Camarilla breakout (L3 for long, H3 for short).
- Signal size: 0.20 discrete to minimize fee churn.
- Session filter: 08-20 UTC to avoid low-volume periods.
- Camarilla levels provide intraday support/resistance; EMA34 filters regime; volume spike confirms breakout conviction.
- Works in bull (breakouts with trend) and bear (strong moves after regime shifts).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate 4h EMA(34) for trend
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1h Camarilla levels (H3, L3) from previous day
    # Need daily OHLC from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h3 = np.zeros(len(df_1d))
    camarilla_l3 = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        high = high_1d[i]
        low = low_1d[i]
        close = close_1d[i]
        range_val = high - low
        camarilla_h3[i] = close + range_val * 1.1 / 6
        camarilla_l3[i] = close - range_val * 1.1 / 6
    
    # Align Camarilla levels to 1h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 1h volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours (08-20 UTC)
    hours = prices.index.hour
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA needs 34, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_4h_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter: price vs 4h EMA34
        bull_regime = curr_close > ema_4h_aligned[i]
        bear_regime = curr_close < ema_4h_aligned[i]
        
        # Volume confirmation: 2.0x threshold (tight to reduce trades)
        vol_confirm = curr_volume > 2.0 * vol_ma[i]
        
        if position == 0:
            # Check for entry signals
            # Long: price breaks above Camarilla H3 in bull regime with volume confirmation
            if curr_close > camarilla_h3_aligned[i] and bull_regime and vol_confirm:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla L3 in bear regime with volume confirmation
            elif curr_close < camarilla_l3_aligned[i] and bear_regime and vol_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long position: exit on break below Camarilla L3
            if curr_close < camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position: exit on break above Camarilla H3
            if curr_close > camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hEMA34_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0