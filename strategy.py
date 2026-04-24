#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1w trend filter and volume confirmation.
- Primary timeframe: 12h for lower trade frequency and reduced fee drag.
- HTF: 1w EMA34 for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Williams Alligator: Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA).
- Entry: Long when Lips > Teeth > Jaw (bullish alignment) AND 1w EMA34 bullish AND volume > 1.5 * 20-period volume MA.
         Short when Lips < Teeth < Jaw (bearish alignment) AND 1w EMA34 bearish AND volume > 1.5 * 20-period volume MA.
- Exit: Opposite Alligator alignment (Lips crosses Teeth) or loss of volume confirmation.
- Signal size: 0.25 discrete to balance profit potential and drawdown control.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
Williams Alligator identifies trend initiation and alignment, while 1w EMA34 filters for higher-timeframe trend.
Volume confirmation ensures institutional participation. Works in bull markets by catching trends early and
in bear markets by identifying downtrends. Low trade frequency minimizes fee drag impact.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator components (SMMA = smoothed moving average)
    def smma(data, period):
        """Smoothed Moving Average - equivalent to EMA with alpha=1/period"""
        if len(data) < period:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (prev_smma * (period-1) + current_price) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Alligator components: Jaw (13), Teeth (8), Lips (5)
    jaw = smma(close, 13)   # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    df_1w_close = df_1w['close'].values
    ema_1w = pd.Series(df_1w_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period 1w volume MA
    df_1w_volume = df_1w['volume'].values
    vol_ma_1w = pd.Series(df_1w_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # Volume confirmation: current 12h volume > 1.5 * 20-period 1w volume MA (aligned)
    volume_spike = volume > (1.5 * vol_ma_1w_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 13)  # Need enough bars for EMA34, volume MA, and Alligator jaw
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator signals
        lips_above_teeth = lips[i] > teeth[i]
        teeth_above_jaw = teeth[i] > jaw[i]
        bullish_alignment = lips_above_teeth and teeth_above_jaw  # Lips > Teeth > Jaw
        
        lips_below_teeth = lips[i] < teeth[i]
        teeth_below_jaw = teeth[i] < jaw[i]
        bearish_alignment = lips_below_teeth and teeth_below_jaw  # Lips < Teeth < Jaw
        
        ema_val = ema_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if vol_spike:
                # Bullish: Alligator bullish alignment AND 1w EMA34 bullish (close > EMA)
                if bullish_alignment and close[i] > ema_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Alligator bearish alignment AND 1w EMA34 bearish (close < EMA)
                elif bearish_alignment and close[i] < ema_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Alligator loses bullish alignment (Lips crosses below Teeth) OR loss of volume confirmation
            if not (lips[i] > teeth[i]) or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator loses bearish alignment (Lips crosses above Teeth) OR loss of volume confirmation
            if not (lips[i] < teeth[i]) or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1wEMA34_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0