#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation.
# Long when price > Alligator Jaw AND Alligator Teeth > Alligator Lips (bullish alignment) AND 1w close > EMA50 AND volume > 1.5x 20-bar average.
# Short when price < Alligator Jaw AND Alligator Teeth < Alligator Lips (bearish alignment) AND 1w close < EMA50 AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 30-100 total trades over 4 years (7-25/year).
# Williams Alligator identifies trend alignment via smoothed medians, EMA50 filters higher-timeframe trend, volume confirms momentum.
# Primary timeframe: 1d, HTF: 1w for EMA trend filter.

name = "1d_WilliamsAlligator_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator (Smoothed Medians)
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    # Lips: 5-period SMMA, shifted 3 bars forward
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    def smma(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (Prev_SMMA * (period-1) + Current_Price) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Calculate Alligator components
    jaw = smma(high, 13)  # Using high for Jaw (typical Alligator uses median price, but high/low works for alignment)
    teeth = smma(high, 8)
    lips = smma(high, 5)
    
    # Apply forward shifts (Jaw: +8, Teeth: +5, Lips: +3)
    jaw_shifted = np.full_like(jaw, np.nan)
    teeth_shifted = np.full_like(teeth, np.nan)
    lips_shifted = np.full_like(lips, np.nan)
    
    if len(jaw) > 8:
        jaw_shifted[8:] = jaw[:-8]
    if len(teeth) > 5:
        teeth_shifted[5:] = teeth[:-5]
    if len(lips) > 3:
        lips_shifted[3:] = lips[:-3]
    
    # Align Alligator components to 1d timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips_shifted)
    
    # 1w EMA50 trend filter
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume confirmation: current 1d volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for Alligator and EMA
    
    for i in range(start_idx, n):
        if np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or \
           np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)  # Volume spike threshold
        
        # Alligator alignment signals
        bullish_alignment = (curr_close > jaw_aligned[i] and 
                           teeth_aligned[i] > lips_aligned[i])
        bearish_alignment = (curr_close < jaw_aligned[i] and 
                           teeth_aligned[i] < lips_aligned[i])
        
        # Trend filter: bullish if close > EMA50, bearish if close < EMA50
        bullish_trend = curr_close > ema_50_aligned[i]
        bearish_trend = curr_close < ema_50_aligned[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: bullish Alligator alignment AND bullish trend AND volume confirmation
            if (bullish_alignment and 
                bullish_trend and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment AND bearish trend AND volume confirmation
            elif (bearish_alignment and 
                  bearish_trend and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: bearish Alligator alignment OR trend turns bearish
            if (bearish_alignment or 
                bearish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: bullish Alligator alignment OR trend turns bullish
            if (bullish_alignment or 
                bullish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals