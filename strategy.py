#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_v1
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA34 trend filter and volume spike confirmation.
- Uses 1h timeframe targeting 60-150 total trades over 4 years (15-37/year)
- Long when price breaks above R1 with volume spike, 4h uptrend (EMA34)
- Short when price breaks below S1 with volume spike, 4h downtrend (EMA34)
- Camarilla levels derived from previous 1d OHLC for structure-aware entries
- Session filter: only trade 08:00-20:00 UTC to avoid low-volume Asian session
- Designed for moderate trade frequency with proven edge on BTC/ETH from historical data
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for filtering (08:00-20:00 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    # Load 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe (wait for completed 1d bar)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Load 4h data ONCE before loop for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA34 for trend filter
    ema34_4h = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Calculate volume spike (20-period volume average on 1h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA34, 20 for volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Session filter: only trade 08:00-20:00 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Skip if any data not ready
        if (np.isnan(ema34_4h_aligned[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Camarilla breakout conditions with volume confirmation and trend filters
        price_above_R1 = close[i] > R1_aligned[i]
        price_below_S1 = close[i] < S1_aligned[i]
        
        # 4h trend filter
        trend_up = close[i] > ema34_4h_aligned[i]
        trend_down = close[i] < ema34_4h_aligned[i]
        
        if position == 0 and in_session:
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
        else:
            # Outside session: flatten position
            signals[i] = 0.0
            position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0