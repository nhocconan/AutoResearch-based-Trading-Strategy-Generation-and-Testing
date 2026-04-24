#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA(34) trend filter and volume confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w EMA(34) for trend filter (defines bull/bear regime on weekly timeframe).
- Entry: Long when price breaks above Donchian(20) high in bull regime with volume > 2.0 * 1d volume MA(20);
         Short when price breaks below Donchian(20) low in bear regime with volume > 2.0 * 1d volume MA(20).
- Exit: Price crosses below Donchian(10) high for long or above Donchian(10) low for short.
- Signal size: 0.25 discrete to balance capture and fee control.
- Donchian breakouts capture momentum; EMA filter avoids counter-trend trades; volume spike confirms conviction.
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
    
    # Get 1d data for volume MA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for EMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA(34)
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 1d volume MA(20) for confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 20)  # EMA needs 34, Donchian needs 20, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Calculate Donchian channels (20-period for entry, 10-period for exit)
        lookback_20 = max(0, i-19)
        donchian_high_20 = np.max(high[lookback_20:i+1])
        donchian_low_20 = np.min(low[lookback_20:i+1])
        
        lookback_10 = max(0, i-9)
        donchian_high_10 = np.max(high[lookback_10:i+1])
        donchian_low_10 = np.min(low[lookback_10:i+1])
        
        # Volume confirmation: 2.0x threshold (strict to reduce trades)
        vol_confirm = curr_volume > 2.0 * vol_ma_1d_aligned[i]
        
        # Trend filter: price relative to 1w EMA
        bull_regime = curr_close > ema_1w_aligned[i]
        bear_regime = curr_close < ema_1w_aligned[i]
        
        if position == 0:
            # Check for entry signals
            # Long: price breaks above Donchian(20) high in bull regime with volume confirmation
            if curr_high > donchian_high_20 and bull_regime and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian(20) low in bear regime with volume confirmation
            elif curr_low < donchian_low_20 and bear_regime and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: exit when price crosses below Donchian(10) high
            if curr_close < donchian_high_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price crosses above Donchian(10) low
            if curr_close > donchian_low_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA34_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0