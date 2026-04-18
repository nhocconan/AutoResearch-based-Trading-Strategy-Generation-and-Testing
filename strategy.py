#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d EMA filter and volume confirmation.
# Williams Alligator uses smoothed SMAs (Jaw, Teeth, Lips) to identify trends.
# When the lines are intertwined (no trend), we stay flat. When they diverge in alignment:
#   - Bull: Lips > Teeth > Jaw (green alignment) -> long
#   - Bear: Jaw > Teeth > Lips (red alignment) -> short
# 1d EMA34 filter ensures we only trade in direction of higher timeframe trend.
# Volume confirmation adds conviction. Designed for low trade frequency (20-40/year).
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).

name = "4h_WilliamsAlligator_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Williams Alligator (smoothed SMAs)
    df_4h = get_htf_data(prices, '4h')
    # Get 1d data for EMA filter
    df_1d = get_htf_data(prices, '1d')
    
    # Williams Alligator: 3 smoothed SMAs
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars
    # Lips: 5-period SMMA, shifted 3 bars
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    
    def smma(data, period):
        """Smoothed Moving Average - equivalent to EMA with alpha=1/period"""
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
                else:
                    result[i] = np.nan
        return result
    
    # Calculate Alligator lines on 4h close
    close_4h = df_4h['close'].values
    
    # Lips: 5-period SMMA, shifted 3
    lips_raw = smma(close_4h, 5)
    lips = np.roll(lips_raw, 3)  # shift right by 3
    lips[:3] = np.nan  # first 3 values invalid after shift
    
    # Teeth: 8-period SMMA, shifted 5
    teeth_raw = smma(close_4h, 8)
    teeth = np.roll(teeth_raw, 5)  # shift right by 5
    teeth[:5] = np.nan
    
    # Jaw: 13-period SMMA, shifted 8
    jaw_raw = smma(close_4h, 13)
    jaw = np.roll(jaw_raw, 8)  # shift right by 8
    jaw[:8] = np.nan
    
    # Align to lower timeframe (4h -> 4h is identity but we use align_htf_to_ltf for consistency)
    # Actually, since we're already on 4h, we can use directly but keep the pattern
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth)
    jaw_aligned = align_htf_to_ltf(prices, df_4h, jaw)
    
    # 1d EMA34 filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 20-period average volume for confirmation (on 4h data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(lips_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(jaw_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        # Alligator alignment
        lips_val = lips_aligned[i]
        teeth_val = teeth_aligned[i]
        jaw_val = jaw_aligned[i]
        
        # Bullish alignment: Lips > Teeth > Jaw
        bullish_alignment = lips_val > teeth_val and teeth_val > jaw_val
        # Bearish alignment: Jaw > Teeth > Lips
        bearish_alignment = jaw_val > teeth_val and teeth_val > lips_val
        
        # EMA filter: price relative to 1d EMA34
        price_above_ema = close[i] > ema_34_aligned[i]
        price_below_ema = close[i] < ema_34_aligned[i]
        
        if position == 0:
            # Long: Bullish alignment AND price above 1d EMA AND volume
            if bullish_alignment and price_above_ema and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment AND price below 1d EMA AND volume
            elif bearish_alignment and price_below_ema and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Alignment breaks (not bullish) OR price crosses below 1d EMA
            if not bullish_alignment or price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alignment breaks (not bearish) OR price crosses above 1d EMA
            if not bearish_alignment or price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals