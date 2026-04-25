#!/usr/bin/env python3
"""
1d_Williams_Alligator_Trend_With_Volume_Filter
Hypothesis: Williams Alligator on daily timeframe identifies trend direction (jaw-teeth-lips alignment).
Enter long when lips > teeth > jaw with volume confirmation (>1.5x 20-period average).
Enter short when lips < teeth < jaw with volume confirmation.
Exit on opposite Alligator alignment or ATR trailing stop (2.5*ATR from extreme).
Alligator acts as dynamic trend filter that reduces whipsaws in ranging markets.
Volume confirmation ensures breakouts have conviction.
Designed for ~30-80 trades over 4 years (7-20/year) via tight Alligator alignment conditions.
Works in both bull (trend following) and bear (short signals) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Alligator calculation (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:  # need 13 for lips (8+5)
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams Alligator components (Smoothed Moving Average - SMMA)
    # Jaw: 13-period SMMA, shifted 8 bars
    jaw_period = 13
    jaw_shift = 8
    # Teeth: 8-period SMMA, shifted 5 bars
    teeth_period = 8
    teeth_shift = 5
    # Lips: 5-period SMMA, shifted 3 bars
    lips_period = 5
    lips_shift = 3
    
    # Calculate SMMA (Smoothed Moving Average)
    def smma(data, period):
        sma = pd.Series(data).rolling(window=period, min_periods=period).mean()
        # SMMA: first value is SMA, then recursive smoothing
        smma_values = np.full_like(data, np.nan, dtype=float)
        if len(sma) >= period:
            smma_values[period-1] = sma.iloc[period-1]
            for i in range(period, len(data)):
                if not np.isnan(sma.iloc[i]):
                    smma_values[i] = (smma_values[i-1] * (period-1) + sma.iloc[i]) / period
        return smma_values
    
    jaw_1d = smma(close_1d, jaw_period)
    teeth_1d = smma(close_1d, teeth_period)
    lips_1d = smma(close_1d, lips_period)
    
    # Apply shifts (Alligator lines are shifted into the future)
    jaw_1d_shifted = np.roll(jaw_1d, jaw_shift)
    teeth_1d_shifted = np.roll(teeth_1d, teeth_shift)
    lips_1d_shifted = np.roll(lips_1d, lips_shift)
    
    # Set shifted values to NaN for the shifted periods
    jaw_1d_shifted[:jaw_shift] = np.nan
    teeth_1d_shifted[:teeth_shift] = np.nan
    lips_1d_shifted[:lips_shift] = np.nan
    
    # Align Alligator components to 1d timeframe (no additional alignment needed as we're on 1d)
    jaw_aligned = jaw_1d_shifted
    teeth_aligned = teeth_1d_shifted
    lips_aligned = lips_1d_shifted
    
    # Get 1w data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # ATR for stoploss (21-period)
    atr_period = 21
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume regime: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0   # highest close since long entry
    short_extreme = 0.0  # lowest close since short entry
    
    # Start index: need warmup for calculations
    start_idx = max(100, atr_period, 20, jaw_period+jaw_shift, teeth_period+teeth_shift, lips_period+lips_shift, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        jaw = jaw_aligned[i]
        teeth = teeth_aligned[i]
        lips = lips_aligned[i]
        ema_trend = ema_50_1w_aligned[i]
        
        # Alligator alignment conditions
        bullish_alignment = lips > teeth and teeth > jaw  # Lips > Teeth > Jaw
        bearish_alignment = lips < teeth and teeth < jaw  # Lips < Teeth < Jaw
        
        if position == 0:
            # Only trade in alignment with 1w trend (EMA50 filter)
            if close[i] > ema_trend:  # 1w uptrend regime
                # Long: bullish Alligator alignment with volume confirmation
                long_signal = bullish_alignment and vol_regime[i]
            else:  # 1w downtrend regime
                # Short: bearish Alligator alignment with volume confirmation
                short_signal = bearish_alignment and vol_regime[i]
            
            if 'long_signal' in locals() and long_signal:
                signals[i] = 0.25
                position = 1
                long_extreme = close[i]
            elif 'short_signal' in locals() and short_signal:
                signals[i] = -0.25
                position = -1
                short_extreme = close[i]
            else:
                signals[i] = 0.0
                # Clear signal variables for next iteration
                if 'long_signal' in locals(): del long_signal
                if 'short_signal' in locals(): del short_signal
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Update extreme for trailing stop
            if close[i] > long_extreme:
                long_extreme = close[i]
            # Exit conditions: 
            # 1. ATR trailing stop (2.5*ATR from extreme)
            atr_stop = long_extreme - 2.5 * atr[i]
            # 2. Alligator turns bearish (lips < teeth)
            if close[i] <= atr_stop or lips_aligned[i] < teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Update extreme for trailing stop
            if close[i] < short_extreme:
                short_extreme = close[i]
            # Exit conditions:
            # 1. ATR trailing stop (2.5*ATR from extreme)
            atr_stop = short_extreme + 2.5 * atr[i]
            # 2. Alligator turns bullish (lips > teeth)
            if close[i] >= atr_stop or lips_aligned[i] > teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Williams_Alligator_Trend_With_Volume_Filter"
timeframe = "1d"
leverage = 1.0