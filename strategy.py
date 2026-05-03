#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator breakout with 1d trend filter and volume confirmation.
# Long when price breaks above Alligator's Jaw (TEMA13) in 1d uptrend with volume spike.
# Short when price breaks below Alligator's Jaw in 1d downtrend with volume spike.
# Uses ATR-based stoploss (signal→0 when price moves against position by 2.0*ATR).
# Designed for 12h timeframe to minimize trade frequency (target: 50-150 total trades over 4 years).
# Williams Alligator identifies trend direction and strength; 1d EMA34 ensures higher timeframe alignment;
# Volume spike confirms institutional interest. Works in both bull and bear markets by only trading
# with the 1d trend, avoiding counter-trend whipsaws.

name = "12h_WilliamsAlligator_1dEMA34_VolumeSpike_ATR"
timeframe = "12h"
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
    
    # Get 12h data for Williams Alligator calculation
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator lines (all using TEMA)
    # Jaw (TEMA13, 8 bars shift) - Blue line
    close_12h = df_12h['close'].values
    ema1 = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean()
    ema2 = ema1.ewm(span=13, adjust=False, min_periods=13).mean()
    ema3 = ema2.ewm(span=13, adjust=False, min_periods=13).mean()
    jaw = (3 * ema1 - 3 * ema2 + ema3).values
    jaw_shifted = np.roll(jaw, 8)
    jaw_shifted[:8] = np.nan  # First 8 values invalid due to shift
    
    # Teeth (TEMA8, 5 bars shift) - Red line
    ema1_t = pd.Series(close_12h).ewm(span=8, adjust=False, min_periods=8).mean()
    ema2_t = ema1_t.ewm(span=8, adjust=False, min_periods=8).mean()
    ema3_t = ema2_t.ewm(span=8, adjust=False, min_periods=8).mean()
    teeth = (3 * ema1_t - 3 * ema2_t + ema3_t).values
    teeth_shifted = np.roll(teeth, 5)
    teeth_shifted[:5] = np.nan  # First 5 values invalid due to shift
    
    # Lips (TEMA5, 3 bars shift) - Green line
    ema1_l = pd.Series(close_12h).ewm(span=5, adjust=False, min_periods=5).mean()
    ema2_l = ema1_l.ewm(span=5, adjust=False, min_periods=5).mean()
    ema3_l = ema2_l.ewm(span=5, adjust=False, min_periods=5).mean()
    lips = (3 * ema1_l - 3 * ema2_l + ema3_l).values
    lips_shifted = np.roll(lips, 3)
    lips_shifted[:3] = np.nan  # First 3 values invalid due to shift
    
    # Align Alligator lines to lower timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_shifted)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (20-period volume MA on primary timeframe)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # Track entry price for ATR-based stoploss
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        vol_spike = volume_spike[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        trend_up = close_val > ema_34_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_34_1d_aligned[i]  # 1d downtrend
        # Alligator is bullish when Lips > Teeth > Jaw (green > red > blue)
        alligator_bullish = lips_val > teeth_val and teeth_val > jaw_val
        # Alligator is bearish when Jaw > Teeth > Lips (blue > red > green)
        alligator_bearish = jaw_val > teeth_val and teeth_val > lips_val
        
        if position == 0:
            # Long: price breaks above Alligator's Jaw AND 1d uptrend AND volume spike AND Alligator bullish alignment
            if close_val > jaw_val and trend_up and vol_spike and alligator_bullish:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: price breaks below Alligator's Jaw AND 1d downtrend AND volume spike AND Alligator bearish alignment
            elif close_val < jaw_val and trend_down and vol_spike and alligator_bearish:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            # Stoploss: price moves against position by 2.0*ATR
            if close_val < entry_price - 2.0 * atr[i]:
                exit_signal = True
            # Exit: price breaks below Alligator's Teeth
            elif close_val < teeth_val:
                exit_signal = True
            # Exit: 1d trend changes to downtrend
            elif not trend_up:
                exit_signal = True
            # Exit: Alligator loses bullish alignment
            elif not alligator_bullish:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions for short
            exit_signal = False
            # Stoploss: price moves against position by 2.0*ATR
            if close_val > entry_price + 2.0 * atr[i]:
                exit_signal = True
            # Exit: price breaks above Alligator's Teeth
            elif close_val > teeth_val:
                exit_signal = True
            # Exit: 1d trend changes to uptrend
            elif not trend_down:
                exit_signal = True
            # Exit: Alligator loses bearish alignment
            elif not alligator_bearish:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals