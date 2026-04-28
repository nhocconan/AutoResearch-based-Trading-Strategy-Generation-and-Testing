#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator (Jaw/Teeth/Lips) with 1w EMA50 trend filter and volume confirmation.
# Alligator: Jaw=SMA(13,8), Teeth=SMA(8,5), Lips=SMA(5,3). 
# Enter long when Lips > Teeth > Jaw (bullish alignment) and 1w EMA50 trending up and volume > 2x 20-bar average.
# Enter short when Lips < Teeth < Jaw (bearish alignment) and 1w EMA50 trending down and volume > 2x 20-bar average.
# Exit when Alligator lines converge (|Lips - Jaw| < 0.1 * ATR(14)) or reverse alignment.
# Uses discrete position sizing (0.25) to limit drawdown. Target: 30-100 total trades over 4 years (7-25/year).
# Williams Alligator identifies trending vs ranging markets; 1w EMA50 ensures higher timeframe alignment;
# volume confirmation filters weak signals. Works in both bull (strong uptrends) and bear (strong downtrends).

name = "1d_WilliamsAlligator_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for exit condition
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 1d
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Williams Alligator: Jaw=SMA(13,8), Teeth=SMA(8,5), Lips=SMA(5,3)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Volume confirmation: >2x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 20)  # Ensure sufficient history for indicators (13+8=21)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(atr[i]) or np.isnan(volume_ma_20[i])):
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
        
        # Alligator alignment
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Convergence exit: |Lips - Jaw| < 0.1 * ATR(14)
        lips_jaw_diff = np.abs(lips[i] - jaw[i])
        convergence_exit = lips_jaw_diff < 0.1 * atr[i]
        
        # Price action
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Bullish alignment, EMA50 up, volume confirm
            if bullish_alignment and ema_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: Bearish alignment, EMA50 down, volume confirm
            elif bearish_alignment and ema_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit on convergence or reverse
            if convergence_exit or not bullish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit on convergence or reverse
            if convergence_exit or not bearish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals