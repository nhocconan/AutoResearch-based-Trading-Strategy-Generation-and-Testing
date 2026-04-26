#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume_v1
Hypothesis: 1h Camarilla R1/S1 breakout in direction of 4h trend (EMA34) with 1d volume confirmation.
4h EMA34 defines intermediate trend; breakouts aligned with it have higher follow-through.
1d volume spike filters for institutional participation. Session filter (08-20 UTC) reduces noise.
Discrete sizing (0.20) limits fee drag. Target: 60-150 total trades over 4 years (15-37/year).
Works in bull/bear: 4h EMA34 adapts to regime; volume/avoids false breakouts in ranging markets.
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for HTF EMA34 trend
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA34 for trend filter
    close_4h = pd.Series(df_4h['close'].values)
    ema_34_4h = close_4h.ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d volume MA20 for spike detection
    vol_1d = pd.Series(df_1d['volume'].values)
    vol_ma_20_1d = vol_1d.rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 1h Camarilla levels (based on previous 1h bar's OHLC)
    # For 1h timeframe, Camarilla uses previous bar's range
    camarilla_r1 = high + 1.1 * (high - low) / 12
    camarilla_s1 = low - 1.1 * (high - low) / 12
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for 4h EMA, 20 for 1d volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(camarilla_r1[i]) or
            np.isnan(camarilla_s1[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Volume spike condition: 1d volume > 2.0x 20-period average
        volume_spike = volume[i] > 2.0 * vol_ma_20_1d_aligned[i]
        
        # Camarilla breakout conditions (use current bar's close vs previous bar's levels)
        breakout_above = close[i] > camarilla_r1[i-1]  # Previous bar's R1
        breakout_below = close[i] < camarilla_s1[i-1]  # Previous bar's S1
        
        if breakout_above and volume_spike:
            # Long signal: breakout above Camarilla R1 with volume, above 4h EMA34 (bullish bias)
            if close[i] > ema_34_4h_aligned[i]:
                if position != 1:
                    signals[i] = 0.20
                    position = 1
                else:
                    signals[i] = 0.20
            else:
                # Hold or flatten if not aligned with 4h trend
                if position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = 0.0
                    position = 0
        elif breakout_below and volume_spike:
            # Short signal: breakout below Camarilla S1 with volume, below 4h EMA34 (bearish bias)
            if close[i] < ema_34_4h_aligned[i]:
                if position != -1:
                    signals[i] = -0.20
                    position = -1
                else:
                    signals[i] = -0.20
            else:
                # Hold or flatten if not aligned with 4h trend
                if position == -1:
                    signals[i] = -0.20
                else:
                    signals[i] = 0.0
                    position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume_v1"
timeframe = "1h"
leverage = 1.0