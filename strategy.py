#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_v1
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike confirmation.
- Uses 1h timeframe targeting 60-150 total trades over 4 years (15-37/year)
- Long when price breaks above R1 with volume spike and 4h uptrend (close > EMA50)
- Short when price breaks below S1 with volume spike and 4h downtrend (close < EMA50)
- Camarilla levels derived from previous 4h OHLC for structure-aware entries
- Volume spike confirms institutional participation (1.8x 20-period average)
- 4h trend filter reduces whipsaw in bear markets (2022) and captures major moves
- Session filter: 08-20 UTC to avoid low-liquidity Asian session
- Position size: 0.20 (discrete to minimize fee churn)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) for filter
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for Camarilla levels and trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Camarilla levels from previous 4h bar
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_close_4h = df_4h['close'].values
    prev_high_4h = df_4h['high'].values
    prev_low_4h = df_4h['low'].values
    
    R1 = prev_close_4h + (prev_high_4h - prev_low_4h) * 1.1 / 12
    S1 = prev_close_4h - (prev_high_4h - prev_low_4h) * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe (wait for completed 4h bar)
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1)
    
    # Calculate 4h EMA50 for trend filter
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate volume spike (1.8x 20-period volume average on 1h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA50, 20 for volume MA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            # Outside session: go flat or hold flat
            signals[i] = 0.0
            position = 0
            continue
        
        # Camarilla breakout conditions with volume confirmation and trend filter
        price_above_R1 = close[i] > R1_aligned[i]
        price_below_S1 = close[i] < S1_aligned[i]
        
        # 4h trend filter
        trend_up = close[i] > ema50_4h_aligned[i]
        trend_down = close[i] < ema50_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 AND volume spike AND 4h uptrend
            if price_above_R1 and volume_spike[i] and trend_up:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 AND volume spike AND 4h downtrend
            elif price_below_S1 and volume_spike[i] and trend_down:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: price falls below S1 OR 4h trend turns down
            if price_below_S1 or not trend_up:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: price rises above R1 OR 4h trend turns up
            if price_above_R1 or not trend_down:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0