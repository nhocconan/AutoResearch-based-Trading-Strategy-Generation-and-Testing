#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA34 trend filter and volume confirmation
# Long when Jaw < Teeth < Lips (bullish alignment) AND close > 1w EMA34 AND volume > 1.5x 20-bar avg
# Short when Jaw > Teeth > Lips (bearish alignment) AND close < 1w EMA34 AND volume > 1.5x 20-bar avg
# Exit when Alligator lines cross (trend weakening)
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 10-25 trades/year on 1d.
# Works in bull markets via trend alignment, works in bear via volume spike requirement
# which captures panic climaxes that often precede reversals. 1d timeframe minimizes fee drag.

name = "1d_WilliamsAlligator_1wEMA34_Trend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    # Calculate EMA(34) on 1w close
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA34 to 1d timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Williams Alligator on 1d data
    # Jaw: Blue Line (13-period SMMA, shifted 8 bars)
    # Teeth: Red Line (8-period SMMA, shifted 5 bars)
    # Lips: Green Line (5-period SMMA, shifted 3 bars)
    close_series = pd.Series(close)
    
    # SMMA (Smoothed Moving Average) calculation
    def smma(values, period):
        sma = pd.Series(values).rolling(window=period, min_periods=period).mean()
        # Convert to smoothed moving average
        result = np.full_like(values, np.nan, dtype=float)
        if len(sma) >= period:
            result[period-1] = sma.iloc[period-1]
            for i in range(period, len(values)):
                if not np.isnan(sma.iloc[i]):
                    result[i] = (result[i-1] * (period-1) + sma.iloc[i]) / period
        return result
    
    jaw = smma(close.values, 13)
    teeth = smma(close.values, 8)
    lips = smma(close.values, 5)
    
    # Apply shifts (Jaw shifted 8, Teeth shifted 5, Lips shifted 3)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted positions
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 13, 8, 5, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(jaw_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_trend = ema_34_1w_aligned[i]
        jaw_val = jaw_shifted[i]
        teeth_val = teeth_shifted[i]
        lips_val = lips_shifted[i]
        curr_close = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when Jaw < Teeth < Lips (bullish alignment) AND close > 1w EMA34 AND volume confirmation
            if jaw_val < teeth_val < lips_val and curr_close > ema_trend and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Jaw > Teeth > Lips (bearish alignment) AND close < 1w EMA34 AND volume confirmation
            elif jaw_val > teeth_val > lips_val and curr_close < ema_trend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when Alligator lines cross (trend weakening)
            if jaw_val >= teeth_val or teeth_val >= lips_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when Alligator lines cross (trend weakening)
            if jaw_val <= teeth_val or teeth_val <= lips_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals