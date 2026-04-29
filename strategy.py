#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d EMA50 trend filter and volume spike confirmation
# Williams Alligator consists of three smoothed moving averages (Jaw, Teeth, Lips)
# Jaw (Blue): 13-period SMMA, shifted 8 bars ahead
# Teeth (Red): 8-period SMMA, shifted 5 bars ahead  
# Lips (Green): 5-period SMMA, shifted 3 bars ahead
# Long when Lips > Teeth > Jaw (bullish alignment) AND close > 1d EMA50 AND volume > 2.0x 20-bar avg
# Short when Lips < Teeth < Jaw (bearish alignment) AND close < 1d EMA50 AND volume > 2.0x 20-bar avg
# Exit when alignment breaks (Lips crosses Teeth or Teeth crosses Jaw)
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 12-37 trades/year on 6h.
# Williams Alligator identifies trend direction and potential reversals through convergence/divergence of SMMA lines.
# Volume confirmation ensures signals have conviction, reducing false breakouts during consolidation.
# 1d EMA50 filter ensures alignment with higher timeframe trend for better win rate in both bull and bear markets.

name = "6h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: Three smoothed moving averages (SMMA)
    # SMMA is similar to EMA but with different smoothing factor
    # SMMA(t) = (SMMA(t-1) * (period-1) + close(t)) / period
    close_series = pd.Series(close)
    
    # Jaw: 13-period SMMA
    jaw = np.zeros(n)
    jaw[:] = np.nan
    jaw[12] = close_series.iloc[:13].mean()  # First value is simple average
    for i in range(13, n):
        jaw[i] = (jaw[i-1] * 12 + close[i]) / 13
    
    # Teeth: 8-period SMMA
    teeth = np.zeros(n)
    teeth[:] = np.nan
    teeth[7] = close_series.iloc[:8].mean()  # First value is simple average
    for i in range(8, n):
        teeth[i] = (teeth[i-1] * 7 + close[i]) / 8
    
    # Lips: 5-period SMMA
    lips = np.zeros(n)
    lips[:] = np.nan
    lips[4] = close_series.iloc[:5].mean()  # First value is simple average
    for i in range(5, n):
        lips[i] = (lips[i-1] * 4 + close[i]) / 5
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_trend = ema_50_1d_aligned[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        curr_close = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Bullish alignment: Lips > Teeth > Jaw
            bullish_alignment = lips_val > teeth_val and teeth_val > jaw_val
            # Bearish alignment: Lips < Teeth < Jaw
            bearish_alignment = lips_val < teeth_val and teeth_val < jaw_val
            
            # Long when bullish alignment AND close > 1d EMA50 AND volume confirmation
            if bullish_alignment and curr_close > ema_trend and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when bearish alignment AND close < 1d EMA50 AND volume confirmation
            elif bearish_alignment and curr_close < ema_trend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when bullish alignment breaks
            bullish_alignment = lips_val > teeth_val and teeth_val > jaw_val
            if not bullish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when bearish alignment breaks
            bearish_alignment = lips_val < teeth_val and teeth_val < jaw_val
            if not bearish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals