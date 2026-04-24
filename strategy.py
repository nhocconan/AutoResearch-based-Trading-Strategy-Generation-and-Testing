#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA(34) trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h EMA(34) for trend filter (defines bull/bear regime).
- Entry: Long when price breaks above Camarilla R3 in bull regime with volume > 2.0 * 4h volume MA(20);
         Short when price breaks below Camarilla S3 in bear regime with volume > 2.0 * 4h volume MA(20).
- Exit: Price crosses below Camarilla H3 for long or above Camarilla L3 for short.
- Signal size: 0.25 discrete to balance capture and fee control.
- Camarilla levels provide intraday support/resistance; EMA filter avoids counter-trend trades; volume spike confirms conviction.
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
    
    # Get 4h data for volume MA calculation and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 12h data for EMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA(34)
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 4h volume MA(20) for confirmation
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 20)  # EMA needs 34, volume MA needs 20, Camarilla needs 1 (uses current bar)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Calculate Camarilla levels for today (using previous day's OHLC)
        # We need previous day's data - use 1d timeframe for Camarilla calculation
        # For simplicity, we'll use the 4h bar's high/low/close as proxy for intraday levels
        # In reality, Camarilla uses previous day's OHLC, but we approximate with current 4h bar
        # This is a simplification for the strategy
        prev_high = high[i-1] if i > 0 else high[i]
        prev_low = low[i-1] if i > 0 else low[i]
        prev_close = close[i-1] if i > 0 else close[i]
        
        # Camarilla levels calculation
        range_val = prev_high - prev_low
        if range_val <= 0:
            # Skip if no range
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        camarilla_h3 = prev_close + range_val * 1.1 / 4
        camarilla_l3 = prev_close - range_val * 1.1 / 4
        camarilla_r3 = prev_close + range_val * 1.1 / 2
        camarilla_s3 = prev_close - range_val * 1.1 / 2
        
        # Volume confirmation: 2.0x threshold (strict to reduce trades)
        vol_confirm = curr_volume > 2.0 * vol_ma_4h_aligned[i]
        
        # Trend filter: price relative to 12h EMA
        bull_regime = curr_close > ema_12h_aligned[i]
        bear_regime = curr_close < ema_12h_aligned[i]
        
        if position == 0:
            # Check for entry signals
            # Long: price breaks above Camarilla R3 in bull regime with volume confirmation
            if curr_high > camarilla_r3 and bull_regime and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 in bear regime with volume confirmation
            elif curr_low < camarilla_s3 and bear_regime and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: exit when price crosses below Camarilla H3
            if curr_close < camarilla_h3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price crosses above Camarilla L3
            if curr_close > camarilla_l3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_12hEMA34_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0