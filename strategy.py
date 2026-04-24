#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA(50) trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d EMA(50) for trend filter (defines bull/bear regime).
- Entry: Long when price breaks above Donchian(20) high in bull regime with volume > 2.0 * 4h volume MA(20);
         Short when price breaks below Donchian(20) low in bear regime with volume > 2.0 * 4h volume MA(20).
- Exit: Price crosses back below Donchian(20) midline for long or above midline for short (mean reversion).
- Signal size: 0.25 discrete to balance capture and fee control.
- Donchian captures breakouts; EMA filter avoids counter-trend trades; volume spike confirms conviction.
- Works in bull (buying breakouts in uptrend) and bear (selling breakdowns in downtrend).
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
    
    # Get 4h data for volume MA calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 4h volume MA(20) for confirmation
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # Calculate Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 20)  # EMA needs 50, Donchian needs 20, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        prev_close = close[i-1] if i > 0 else curr_close
        
        # Volume confirmation: 2.0x threshold (strict to reduce trades)
        vol_confirm = curr_volume > 2.0 * vol_ma_4h_aligned[i]
        
        # Trend filter: price relative to 1d EMA
        bull_regime = curr_close > ema_1d_aligned[i]
        bear_regime = curr_close < ema_1d_aligned[i]
        
        if position == 0:
            # Check for entry signals
            # Long: price breaks above Donchian(20) high in bull regime with volume confirmation
            if (curr_high > highest_high[i] and prev_close <= highest_high[i] and 
                bull_regime and vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian(20) low in bear regime with volume confirmation
            elif (curr_low < lowest_low[i] and prev_close >= lowest_low[i] and 
                  bear_regime and vol_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: exit when price crosses back below Donchian midline
            if prev_close >= donchian_mid[i] and curr_close < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price crosses back above Donchian midline
            if prev_close <= donchian_mid[i] and curr_close > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0