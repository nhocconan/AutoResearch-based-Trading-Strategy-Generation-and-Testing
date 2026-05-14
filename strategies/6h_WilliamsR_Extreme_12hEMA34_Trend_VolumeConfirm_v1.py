#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R extreme with 12h EMA trend filter and volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 12h EMA(34) for trend filter (defines bull/bear regime).
- Entry: Long when Williams %R(14) crosses above -20 from below in bull regime with volume > 2.0 * 6h volume MA(20);
         Short when Williams %R(14) crosses below -80 from above in bear regime with volume > 2.0 * 6h volume MA(20).
- Exit: Williams %R crosses back below -50 for long or above -50 for short (mean reversion in extreme zones).
- Signal size: 0.25 discrete to balance capture and fee control.
- Williams %R captures overbought/oversold extremes; EMA filter avoids counter-trend trades; volume spike confirms conviction.
- Works in bull (buying dips in uptrend) and bear (selling rallies in downtrend).
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
    
    # Get 6h data for Williams %R calculation and volume
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    # Get 12h data for EMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA(34)
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 6h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 6h volume MA(20) for confirmation
    volume_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 14, 20)  # EMA needs 34, Williams %R needs 14, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_6h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        prev_williams_r = williams_r[i-1] if i > 0 else -50
        
        # Volume confirmation: 2.0x threshold (strict to reduce trades)
        vol_confirm = curr_volume > 2.0 * vol_ma_6h_aligned[i]
        
        # Trend filter: price relative to 12h EMA
        bull_regime = curr_close > ema_12h_aligned[i]
        bear_regime = curr_close < ema_12h_aligned[i]
        
        if position == 0:
            # Check for entry signals
            # Long: Williams %R crosses above -20 from below in bull regime with volume confirmation
            if (prev_williams_r <= -20 and williams_r[i] > -20 and 
                bull_regime and vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -80 from above in bear regime with volume confirmation
            elif (prev_williams_r >= -80 and williams_r[i] < -80 and 
                  bear_regime and vol_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: exit when Williams %R crosses below -50
            if prev_williams_r >= -50 and williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when Williams %R crosses above -50
            if prev_williams_r <= -50 and williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_12hEMA34_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0