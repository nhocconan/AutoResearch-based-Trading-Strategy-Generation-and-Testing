#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator + 1d EMA50 trend filter + volume spike confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) SMAs on median price.
- Entry: Long when Alligator aligned bullish (Lips > Teeth > Jaw) AND price > Lips AND 1d EMA50 bullish AND volume > 2.0 * volume MA(30).
         Short when Alligator aligned bearish (Lips < Teeth < Jaw) AND price < Lips AND 1d EMA50 bearish AND volume > 2.0 * volume MA(30).
- Exit: Close-based reversal - exit long when Alligator alignment turns bearish OR price < Jaw,
        exit short when Alligator alignment turns bullish OR price > Jaw.
- Signal size: 0.25 discrete to balance return and drawdown.
Uses 1d EMA50 trend filter and Williams Alligator for multi-timeframe confluence in BTC/ETH/SOL.
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Williams Alligator on 12h data
    # Jaw: 13-period SMA, shifted 8 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan
    
    # Teeth: 8-period SMA, shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    
    # Lips: 5-period SMA, shifted 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
    # Calculate volume MA(30) for confirmation
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 30)  # Need enough bars for Alligator and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(lips[i]) or np.isnan(teeth[i]) or 
            np.isnan(jaw[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (2.0x threshold)
            vol_confirmed = curr_volume > 2.0 * vol_ma[i]
            
            # Alligator aligned bullish: Lips > Teeth > Jaw
            bullish_aligned = lips[i] > teeth[i] and teeth[i] > jaw[i]
            # Alligator aligned bearish: Lips < Teeth < Jaw
            bearish_aligned = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            # Long: Bullish Alligator AND price > Lips AND 1d EMA50 bullish AND volume confirmed
            if bullish_aligned and curr_close > lips[i] and curr_close > ema_1d_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator AND price < Lips AND 1d EMA50 bearish AND volume confirmed
            elif bearish_aligned and curr_close < lips[i] and curr_close < ema_1d_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when Alligator turns bearish OR price < Jaw
            bearish_aligned = lips[i] < teeth[i] and teeth[i] < jaw[i]
            if bearish_aligned or curr_close < jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when Alligator turns bullish OR price > Jaw
            bullish_aligned = lips[i] > teeth[i] and teeth[i] > jaw[i]
            if bullish_aligned or curr_close > jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0