#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator with 12h EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) SMAs on median price.
- Entry: Long when Alligator is bullish (Lips > Teeth > Jaw) AND price > Lips AND 12h EMA50 bullish AND volume > 2.0 * volume MA(20).
         Short when Alligator is bearish (Lips < Teeth < Jaw) AND price < Lips AND 12h EMA50 bearish AND volume > 2.0 * volume MA(20).
- Exit: Close-based reversal - exit long when price crosses below Teeth,
        exit short when price crosses above Teeth.
- Signal size: 0.25 discrete to balance profit potential and drawdown control.
Designed to work in both bull and bear markets via trend filter and mean-reversion exits.
Williams Alligator catches emerging trends while avoiding whipsaws in ranging markets.
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
    
    # Calculate median price for Alligator
    median_price = (high + low) / 2.0
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    df_12h_close = df_12h['close'].values
    ema_12h = pd.Series(df_12h_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams Alligator components (using 4h median price)
    # Jaw: 13-period SMMA, 8 periods ahead
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMMA, 5 periods ahead
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMMA, 3 periods ahead
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align HTF indicators to 4h
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50, 13, 8, 5, 20)  # Need enough bars for EMA50, Alligator, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (2.0x threshold)
            vol_confirmed = curr_volume > 2.0 * vol_ma[i]
            
            # Alligator bullish: Lips > Teeth > Jaw
            alligator_bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
            # Alligator bearish: Lips < Teeth < Jaw
            alligator_bearish = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            # Long: Alligator bullish AND price > Lips AND 12h EMA50 bullish AND volume confirmed
            if alligator_bullish and curr_close > lips[i] and curr_close > ema_12h_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish AND price < Lips AND 12h EMA50 bearish AND volume confirmed
            elif alligator_bearish and curr_close < lips[i] and curr_close < ema_12h_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below Teeth (mean reversion)
            if curr_close < teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above Teeth (mean reversion)
            if curr_close > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_Alligator_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0