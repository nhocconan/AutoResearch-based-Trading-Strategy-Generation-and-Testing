#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w trend filter and volume confirmation.
# Williams Alligator consists of three smoothed moving averages (Jaw, Teeth, Lips).
# Long when Lips > Teeth > Jaw (bullish alignment) with 1w EMA50 uptrend and volume > 1.3x 20-bar average.
# Short when Lips < Teeth < Jaw (bearish alignment) with 1w EMA50 downtrend and volume > 1.3x 20-bar average.
# Exit when Alligator lines cross (Lips crosses Teeth) or price touches the opposite jaw line.
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 30-80 total trades over 4 years (7-20/year).
# Williams Alligator identifies trend phases; 1w EMA50 ensures higher timeframe alignment;
# volume confirmation filters weak signals. Works in both bull (trend following) and bear (trend continuation).

name = "1d_WilliamsAlligator_1wEMA50_Trend_VolumeConfirm_v1"
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 1d
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Williams Alligator: three smoothed moving averages
    # Jaw: 13-period SMMA, shifted by 8 bars
    # Teeth: 8-period SMMA, shifted by 5 bars
    # Lips: 5-period SMMA, shifted by 3 bars
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    
    def smma(data, period):
        """Smoothed Moving Average"""
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_PRICE) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Apply shifts (Jaw shifted by 8, Teeth by 5, Lips by 3)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set shifted values to NaN for invalid positions
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Volume confirmation: >1.3x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.3 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient history for volume MA and Alligator
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(volume_ma_20[i]) or 
            np.isnan(lips_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(jaw_shifted[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1w EMA50 trend: slope over 3 periods
        if i >= 3:
            ema_slope = (ema_50_aligned[i] - ema_50_aligned[i-3]) / 3
            ema_trend_up = ema_slope > 0
            ema_trend_down = ema_slope < 0
        else:
            ema_trend_up = False
            ema_trend_down = False
        
        # Alligator conditions
        lips_val = lips_shifted[i]
        teeth_val = teeth_shifted[i]
        jaw_val = jaw_shifted[i]
        
        # Bullish alignment: Lips > Teeth > Jaw
        bullish_align = lips_val > teeth_val > jaw_val
        # Bearish alignment: Lips < Teeth < Jaw
        bearish_align = lips_val < teeth_val < jaw_val
        
        # Exit conditions: Lips crosses Teeth or price touches opposite jaw
        lips_cross_teeth = (position == 1 and lips_val <= teeth_val) or (position == -1 and lips_val >= teeth_val)
        price_touch_jaw = (position == 1 and low[i] <= jaw_val) or (position == -1 and high[i] >= jaw_val)
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: bullish alignment, EMA50 up, volume confirm
            if bullish_align and ema_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish alignment, EMA50 down, volume confirm
            elif bearish_align and ema_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit
            if lips_cross_teeth or price_touch_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit
            if lips_cross_teeth or price_touch_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals