#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator + 1w Elder Ray Power + volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w Elder Ray Power (bull/bear power) for trend filter (bullish if bull_power > 0, bearish if bear_power < 0).
- Williams Alligator: Jaw(13), Teeth(8), Lips(5) SMAs on median price. Long when Lips > Teeth > Jaw (bullish alignment), short when Lips < Teeth < Jaw (bearish alignment).
- Volume confirmation: current volume > 1.5 * 20-period volume MA to filter weak signals.
- ATR-based stoploss: exit when price moves against position by 2.5 * ATR(14) (using 12h ATR).
- Signal size: 0.25 discrete to minimize fee churn and control drawdown.
Designed to catch strong trends with multi-timeframe alignment and volume filter, works in both bull and bear markets by requiring HTF trend alignment.
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
    
    # Get 12h data for ATR and Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need for ATR and Alligator
        return np.zeros(n)
    
    # Get 1w data for Elder Ray Power
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w Elder Ray Power: Bull Power = High - EMA13, Bear Power = Low - EMA13
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    ema_13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1w = high_1w - ema_13_1w
    bear_power_1w = low_1w - ema_13_1w
    bull_power_1w_aligned = align_htf_to_ltf(prices, df_1w, bull_power_1w)
    bear_power_1w_aligned = align_htf_to_ltf(prices, df_1w, bear_power_1w)
    
    # Calculate 12h ATR(14) for stoploss
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12h volume MA(20) for confirmation
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Williams Alligator on 12h: Jaw(13), Teeth(8), Lips(5) SMAs of median price
    median_price_12h = (high_12h + low_12h) / 2
    jaw_12h = pd.Series(median_price_12h).rolling(window=13, min_periods=13).mean().values
    teeth_12h = pd.Series(median_price_12h).rolling(window=8, min_periods=8).mean().values
    lips_12h = pd.Series(median_price_12h).rolling(window=5, min_periods=5).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(100, 30, 20, 14, 20, 13, 8, 5)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power_1w_aligned[i]) or np.isnan(bear_power_1w_aligned[i]) or 
            np.isnan(atr_12h[i]) or np.isnan(vol_ma_12h[i]) or 
            np.isnan(jaw_12h[i]) or np.isnan(teeth_12h[i]) or np.isnan(lips_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold)
            vol_confirmed = curr_volume > 1.5 * vol_ma_12h[i]
            
            # Determine 1w trend: bullish if bull_power > 0, bearish if bear_power < 0
            trend_bullish = bull_power_1w_aligned[i] > 0
            trend_bearish = bear_power_1w_aligned[i] < 0
            
            # Williams Alligator alignment
            alligator_bullish = lips_12h[i] > teeth_12h[i] and teeth_12h[i] > jaw_12h[i]
            alligator_bearish = lips_12h[i] < teeth_12h[i] and teeth_12h[i] < jaw_12h[i]
            
            # Long: Alligator bullish AND 1w trend bullish AND volume confirmed
            if alligator_bullish and trend_bullish and vol_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Alligator bearish AND 1w trend bearish AND volume confirmed
            elif alligator_bearish and trend_bearish and vol_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: exit on stoploss or Alligator turns bearish (reversal signal)
            stop_loss = entry_price - 2.5 * atr_12h[i]
            if curr_low < stop_loss or not (lips_12h[i] > teeth_12h[i] and teeth_12h[i] > jaw_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on stoploss or Alligator turns bullish (reversal signal)
            stop_loss = entry_price + 2.5 * atr_12h[i]
            if curr_high > stop_loss or not (lips_12h[i] < teeth_12h[i] and teeth_12h[i] < jaw_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_1wElderRay_Power_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0