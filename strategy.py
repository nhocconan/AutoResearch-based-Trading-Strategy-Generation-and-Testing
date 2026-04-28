#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume spike confirmation
# Long when Jaw < Teeth < Lips (bullish alignment) AND close > 1d EMA50 AND volume > 3.0x 20-bar avg
# Short when Jaw > Teeth > Lips (bearish alignment) AND close < 1d EMA50 AND volume > 3.0x 20-bar avg
# Exit when Alligator alignment breaks (Jaw-Teeth-Lips not in proper order) OR close crosses 1d EMA50
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year on 12h.
# Williams Alligator catches trending moves with smoothed SMAs (13,8,5) offset by future bars.
# Volume spike requirement filters for high-conviction moves, reducing whipsaw in ranging markets.
# 12h timeframe naturally limits trade frequency while capturing multi-day trends.

name = "12h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 12h timeframe (using close prices)
    # Jaw: 13-period SMMA, offset by 8 bars
    # Teeth: 8-period SMMA, offset by 5 bars  
    # Lips: 5-period SMMA, offset by 3 bars
    # SMMA = smoothed moving average (similar to Wilder's smoothing)
    
    def smma(values, period):
        """Smoothed Moving Average"""
        if len(values) < period:
            return np.full_like(values, np.nan, dtype=float)
        result = np.full_like(values, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_value) / period
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Apply Alligator offsets (shift right by offset bars)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # First offset bars become NaN
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Volume confirmation: >3.0x 20-bar average volume (very strict for low trade frequency)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 3.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 13)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_trend = ema_50_1d_aligned[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        curr_close = close[i]
        
        # Check Alligator alignment
        bullish_aligned = jaw_val < teeth_val < lips_val  # Jaw < Teeth < Lips
        bearish_aligned = jaw_val > teeth_val > lips_val  # Jaw > Teeth > Lips
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when bullish alignment AND close > 1d EMA50 AND volume confirmation
            if bullish_aligned and curr_close > ema_trend and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when bearish alignment AND close < 1d EMA50 AND volume confirmation
            elif bearish_aligned and curr_close < ema_trend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when alignment breaks or close < EMA50
            if not bullish_aligned or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when alignment breaks or close > EMA50
            if not bearish_aligned or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals