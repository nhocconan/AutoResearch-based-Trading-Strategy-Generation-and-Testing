#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation.
# Long when Alligator jaws (13-period smoothed median) cross above teeth (8-period smoothed median) 
# in 1d uptrend with volume spike (>1.8x 20-period volume MA). Short on opposite crossover 
# in 1d downtrend with volume spike. Uses ATR stoploss (signal→0 when price moves 
# against position by 2.5*ATR). Alligator is trend-following but less whipsaw-prone than 
# standard MA crossovers due to smoothing and phase shifts. Works in bull/bear by only 
# trading with 1d trend, avoiding counter-trend signals. Target 20-50 trades/year.

name = "4h_WilliamsAlligator_1dEMA50_VolumeSpike_ATR"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR for stoploss (using primary timeframe)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 4h data for Alligator calculation
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator components (all using median price)
    median_price = (df_4h['high'] + df_4h['low']) / 2
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    median_4h = (high_4h + low_4h) / 2
    
    # Jaws: 13-period SMMA of median, shifted 8 bars
    jaws_raw = pd.Series(median_4h).rolling(window=13, min_periods=13).mean().values
    jaws = np.roll(jaws_raw, 8)
    jaws[:8] = np.nan  # First 8 values invalid due to shift
    
    # Teeth: 8-period SMMA of median, shifted 5 bars
    teeth_raw = pd.Series(median_4h).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)
    teeth[:5] = np.nan  # First 5 values invalid due to shift
    
    # Lips: 5-period SMMA of median, shifted 3 bars
    lips_raw = pd.Series(median_4h).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)
    lips[:3] = np.nan  # First 3 values invalid due to shift
    
    # Align Alligator components to LTF
    jaws_aligned = align_htf_to_ltf(prices, df_4h, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection (20-period volume MA on primary timeframe)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.8)  # Volume at least 1.8x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # Track entry price for ATR-based stoploss
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        vol_spike = volume_spike[i]
        jaw_val = jaws_aligned[i]
        tooth_val = teeth_aligned[i]
        lip_val = lips_aligned[i]
        trend_up = close_val > ema_50_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_50_1d_aligned[i]  # 1d downtrend
        
        if position == 0:
            # Long: Alligator lines aligned bullish (jaws > teeth > lips) AND 1d uptrend AND volume spike
            if jaw_val > tooth_val and tooth_val > lip_val and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: Alligator lines aligned bearish (jaws < teeth < lips) AND 1d downtrend AND volume spike
            elif jaw_val < tooth_val and tooth_val < lip_val and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            # Stoploss: price moves against position by 2.5*ATR
            if close_val < entry_price - 2.5 * atr[i]:
                exit_signal = True
            # Exit: Alligator lines no longer bullish aligned
            elif not (jaw_val > tooth_val and tooth_val > lip_val):
                exit_signal = True
            # Exit: 1d trend changes to downtrend
            elif not trend_up:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions for short
            exit_signal = False
            # Stoploss: price moves against position by 2.5*ATR
            if close_val > entry_price + 2.5 * atr[i]:
                exit_signal = True
            # Exit: Alligator lines no longer bearish aligned
            elif not (jaw_val < tooth_val and tooth_val < lip_val):
                exit_signal = True
            # Exit: 1d trend changes to uptrend
            elif not trend_down:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals