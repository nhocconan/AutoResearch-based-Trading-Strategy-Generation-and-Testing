#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h Supertrend(10,3) trend filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h Supertrend for trend direction (bullish when Supertrend is below price, bearish when above).
- Entry: Price breaks above/below 4h Donchian(20) levels with volume > 1.8 * 20-period volume MA and HTF Supertrend alignment.
- Exit: ATR-based stoploss (2.5 * ATR(14)) or Donchian level reversal (opposite channel touch).
- Signal size: 0.25 discrete to minimize fee churn and control drawdown.
Designed to capture strong 4h momentum moves with volume confirmation and trend filtering.
Supertrend provides adaptive trend following that works in both bull and bear markets.
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
    
    # Get 4h data for Donchian levels, ATR, and volume MA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # Need for Donchian and volume MA
        return np.zeros(n)
    
    # Get 12h data for Supertrend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Calculate 12h Supertrend(10,3)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # ATR calculation for Supertrend
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high_12h + low_12h) / 2
    upper_band = hl2 + (3 * atr_12h)
    lower_band = hl2 - (3 * atr_12h)
    
    upper_band = pd.Series(upper_band)
    lower_band = pd.Series(lower_band)
    
    for i in range(1, len(upper_band)):
        if close_12h[i-1] <= upper_band.iloc[i-1]:
            upper_band.iloc[i] = min(upper_band.iloc[i], upper_band.iloc[i-1])
        else:
            upper_band.iloc[i] = upper_band.iloc[i]
            
        if close_12h[i-1] >= lower_band.iloc[i-1]:
            lower_band.iloc[i] = max(lower_band.iloc[i], lower_band.iloc[i-1])
        else:
            lower_band.iloc[i] = lower_band.iloc[i]
    
    supertrend = np.zeros(len(close_12h))
    for i in range(len(close_12h)):
        if i == 0:
            supertrend[i] = upper_band.iloc[i]
        elif close_12h[i-1] <= supertrend[i-1]:
            supertrend[i] = upper_band.iloc[i]
        else:
            supertrend[i] = lower_band.iloc[i]
    
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    
    # Calculate 4h Donchian(20) levels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 10, 20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(supertrend_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.8x threshold)
            vol_confirmed = curr_volume > 1.8 * vol_ma[i]
            
            # Determine 12h Supertrend: bullish if price > Supertrend, bearish if price < Supertrend
            htf_close_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
            htf_close = htf_close_aligned[i]
            
            trend_bullish = htf_close > supertrend_aligned[i]
            trend_bearish = htf_close < supertrend_aligned[i]
            
            # Long: price breaks above Donchian upper AND 12h trend bullish AND volume confirmed
            if curr_high > high_roll[i] and trend_bullish and vol_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below Donchian lower AND 12h trend bearish AND volume confirmed
            elif curr_low < low_roll[i] and trend_bearish and vol_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: exit on stoploss or price breaks below Donchian lower (reversal signal)
            stop_loss = entry_price - 2.5 * atr[i]
            if curr_low < stop_loss or curr_low < low_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on stoploss or price breaks above Donchian upper (reversal signal)
            stop_loss = entry_price + 2.5 * atr[i]
            if curr_high > stop_loss or curr_high > high_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hSupertrend10_3_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0