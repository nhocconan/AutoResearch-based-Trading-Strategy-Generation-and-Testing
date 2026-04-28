#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d Trend Filter + Volume Spike
# Williams Alligator: Jaw (EMA13, 8-bar shift), Teeth (EMA8, 5-bar shift), Lips (EMA5, 3-bar shift)
# Long when Lips > Teeth > Jaw (bullish alignment) and price > 1d EMA50 (uptrend) and volume > 2.0x 20-bar average
# Short when Lips < Teeth < Jaw (bearish alignment) and price < 1d EMA50 (downtrend) and volume confirmation
# Exit when Alligator alignment breaks (Lips crosses Teeth or Teeth crosses Jaw)
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 50-150 total trades over 4 years (12-37/year).
# Alligator identifies trend via smoothed MAs with future shifts. Works in both bull/bear markets
# by requiring alignment with higher-timeframe trend (1d EMA50). Volume confirmation filters weak signals.

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
    
    # Calculate 12h SMAs for Williams Alligator
    close_series = pd.Series(close)
    sma_5 = close_series.rolling(window=5, min_periods=5).mean().values   # Lips
    sma_8 = close_series.rolling(window=8, min_periods=8).mean().values   # Teeth
    sma_13 = close_series.rolling(window=13, min_periods=13).mean().values # Jaw
    
    # Williams Alligator with shifts (Jaw: shift 8, Teeth: shift 5, Lips: shift 3)
    jaw = np.roll(sma_13, 8)   # EMA13, 8-bar shift
    teeth = np.roll(sma_8, 5)  # EMA8, 5-bar shift
    lips = np.roll(sma_5, 3)   # EMA5, 3-bar shift
    
    # Fill NaN from rolling and rolling shifts
    jaw[:13+8] = np.nan
    teeth[:8+5] = np.nan
    lips[:5+3] = np.nan
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:  # Need sufficient data for EMA calculation
        return np.zeros(n)
    
    # Calculate 1d EMA (50-period)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 12h
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13+8, 8+5, 5+3)  # Ensure sufficient history
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1d EMA trend filter: price > EMA50 = uptrend, price < EMA50 = downtrend
        ema_trend_up = close[i] > ema_50_1d_aligned[i]
        ema_trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Williams Alligator alignment
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Bullish alignment, price > 1d EMA50 (uptrend), volume confirm
            if bullish_alignment and ema_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: Bearish alignment, price < 1d EMA50 (downtrend), volume confirm
            elif bearish_alignment and ema_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit when bullish alignment breaks
            if not bullish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit when bearish alignment breaks
            if not bearish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals