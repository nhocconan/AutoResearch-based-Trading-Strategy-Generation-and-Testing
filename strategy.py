#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator system with 1d trend filter and volume spike
# Williams Alligator uses three smoothed moving averages (Jaw, Teeth, Lips) to identify trends
# Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA50 AND volume > 1.5x 20-bar avg
# Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA50 AND volume > 1.5x 20-bar avg
# Exit when Alligator lines cross (trend weakness) or price crosses 1d EMA50
# Uses discrete sizing 0.25 to minimize fee drag. Target: 15-35 trades/year on 12h.
# Alligator filters whipsaws in ranging markets while capturing strong trends.
# Works in bull via upward alignments, works in bear via downward alignments with volume confirmation.

name = "12h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
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
    
    # Align 1d EMA50 to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: three smoothed moving averages
    # Jaw: 13-period SMMA smoothed by 8 periods
    # Teeth: 8-period SMMA smoothed by 5 periods  
    # Lips: 5-period SMMA smoothed by 3 periods
    def smoothed_mma(data, period):
        """Smoothed Moving Average (SMMA) - Wilder's smoothing"""
        if len(data) < period:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (Prev SMMA*(period-1) + Current Price) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Calculate Alligator components
    jaw = smoothed_mma(close, 13)  # 13-period SMMA
    jaw = smoothed_mma(jaw, 8)     # smoothed by 8 periods
    
    teeth = smoothed_mma(close, 8)   # 8-period SMMA
    teeth = smoothed_mma(teeth, 5)   # smoothed by 5 periods
    
    lips = smoothed_mma(close, 5)    # 5-period SMMA
    lips = smoothed_mma(lips, 3)     # smoothed by 3 periods
    
    # Volume confirmation: >1.5x 20-bar average volume (balanced filter for appropriate trade frequency)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(lips[i]) or np.isnan(teeth[i]) or 
            np.isnan(jaw[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_trend = ema_50_1d_aligned[i]
        lips_val = lips[i]
        teeth_val = teeth[i]
        jaw_val = jaw[i]
        curr_close = close[i]
        prev_close = close[i-1]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Bullish Alligator alignment: Lips > Teeth > Jaw
            bullish_align = lips_val > teeth_val and teeth_val > jaw_val
            # Bearish Alligator alignment: Lips < Teeth < Jaw
            bearish_align = lips_val < teeth_val and teeth_val < jaw_val
            
            # Long when bullish alignment AND price > 1d EMA50 AND volume confirmation
            if bullish_align and curr_close > ema_trend and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when bearish alignment AND price < 1d EMA50 AND volume confirmation
            elif bearish_align and curr_close < ema_trend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when Alligator loses bullish alignment or price < EMA50
            bullish_align = lips_val > teeth_val and teeth_val > jaw_val
            if not bullish_align or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when Alligator loses bearish alignment or price > EMA50
            bearish_align = lips_val < teeth_val and teeth_val < jaw_val
            if not bearish_align or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals