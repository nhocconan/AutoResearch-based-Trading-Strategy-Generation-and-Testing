#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3L3 breakout with 4h EMA34 trend filter and volume confirmation.
- Camarilla H3/L3 on 1h chart identifies key intraday support/resistance levels.
- 4h EMA34 provides higher-timeframe trend filter to align with momentum and reduce counter-trend trades.
- Volume spike (>2.0x 24-period average) confirms breakout validity and reduces false signals.
- Session filter (08-20 UTC) avoids low-liquidity periods.
- Discrete position sizing (0.20) minimizes fee churn while allowing meaningful returns.
- Target trades: 60-150 total over 4 years = 15-37/year for 1h timeframe to avoid fee drag.
- Works in bull/bear markets via 4h trend filter and volatility-based volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1h data ONCE before loop for Camarilla calculation
    # We'll use 1h data to calculate Camarilla levels (standard 1h OHLC)
    # But we need to ensure we're using completed 1h bars for calculation
    # Since prices is already 1h, we can calculate directly
    
    # Calculate Camarilla levels on 1h using previous completed bar
    # H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    # where C, H, L are from previous completed 1h bar
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan  # First bar has no previous
    
    camarilla_high = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_low = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Get 4h data ONCE before loop for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # 4h EMA34 trend filter
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Volume confirmation: > 2.0x 24-period average volume
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * vol_ma
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(24, 1) + 1  # volume MA needs 24, Camarilla needs 1 previous bar
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(camarilla_high[i]) or np.isnan(camarilla_low[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Camarilla H3 with volume spike and above 4h EMA34 (bullish higher-timeframe trend)
            if close[i] > camarilla_high[i] and volume_spike[i] and close[i] > ema_34_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: break below Camarilla L3 with volume spike and below 4h EMA34 (bearish higher-timeframe trend)
            elif close[i] < camarilla_low[i] and volume_spike[i] and close[i] < ema_34_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price closes below Camarilla L3 OR below 4h EMA34 (trend change)
            if close[i] < camarilla_low[i] or close[i] < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price closes above Camarilla H3 OR above 4h EMA34 (trend change)
            if close[i] > camarilla_high[i] or close[i] > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hEMA34_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0