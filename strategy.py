#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + 1d EMA50 trend filter + volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA50 for trend filter (price above/below EMA50 defines bull/bear regime).
- Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) SMAs on median price.
- Entry: Long when Lips > Teeth > Jaw (bullish alignment) in bull regime with volume > 1.5 * 6h volume MA(20);
         Short when Lips < Teeth < Jaw (bearish alignment) in bear regime with volume > 1.5 * 6h volume MA(20).
- Exit: Opposite Alligator alignment (Lips crosses Teeth) or ATR trailing stop (2.0 * ATR(14)).
- Signal size: 0.25 discrete to balance capture and fee control.
- Designed for BTC/ETH: Alligator identifies trend phases, EMA50 filter avoids counter-trend trades,
  volume confirmation ensures strong participation. Works in bull (trend continuation) and bear (strong moves after exhaustion).
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
    
    # Get 6h data for Williams Alligator and volume
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h volume MA(20) for confirmation
    volume_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    # Calculate 6h ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Williams Alligator on 6h data
    # Median price = (high + low) / 2
    median_price = (high + low) / 2
    
    # Jaw: 13-period SMA, smoothed by 8 periods
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean()
    jaw = jaw.rolling(window=8, min_periods=8).mean().values
    
    # Teeth: 8-period SMA, smoothed by 5 periods
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean()
    teeth = teeth.rolling(window=5, min_periods=5).mean().values
    
    # Lips: 5-period SMA, smoothed by 3 periods
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean()
    lips = lips.rolling(window=3, min_periods=3).mean().values
    
    # Align Alligator lines to LTF
    jaw_aligned = align_htf_to_ltf(prices, df_6h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14, 13+8, 8+5, 5+3)  # EMA50 needs 50, volume MA needs 20, ATR needs 14, Alligator needs smoothing
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_6h_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume spike confirmation: 1.5x threshold (balanced to avoid overtrading)
        vol_spike = curr_volume > 1.5 * vol_ma_6h_aligned[i]
        
        # Trend filter: price above/below 1d EMA50
        bull_regime = curr_close > ema_50_1d_aligned[i]
        bear_regime = curr_close < ema_50_1d_aligned[i]
        
        # Alligator alignment
        lips_above_teeth = lips_aligned[i] > teeth_aligned[i]
        teeth_above_jaw = teeth_aligned[i] > jaw_aligned[i]
        lips_below_teeth = lips_aligned[i] < teeth_aligned[i]
        teeth_below_jaw = teeth_aligned[i] < jaw_aligned[i]
        
        bullish_alignment = lips_above_teeth and teeth_above_jaw
        bearish_alignment = lips_below_teeth and teeth_below_jaw
        
        if position == 0:
            # Check for entry signals
            # Long: bullish alignment in bull regime with volume spike
            if bullish_alignment and bull_regime and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment in bear regime with volume spike
            elif bearish_alignment and bear_regime and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: exit on bearish alignment or ATR trailing stop
            if bearish_alignment or curr_low <= high_since_entry - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                # Update highest high since entry for trailing stop
                if 'high_since_entry' not in locals():
                    high_since_entry = curr_close
                else:
                    high_since_entry = max(high_since_entry, curr_close)
        elif position == -1:
            # Short position: exit on bullish alignment or ATR trailing stop
            if bullish_alignment or curr_high >= low_since_entry + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                # Update lowest low since entry for trailing stop
                if 'low_since_entry' not in locals():
                    low_since_entry = curr_close
                else:
                    low_since_entry = min(low_since_entry, curr_close)
    
    return signals

name = "6h_WilliamsAlligator_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0