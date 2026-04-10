#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d volume spike regime filter
# - Long when 12h price > Alligator Jaw (TEMA13) AND 1d volume > 2.0x 20-period volume SMA
# - Short when 12h price < Alligator Lips (TEMA8) AND 1d volume > 2.0x 20-period volume SMA
# - Exit: price crosses Alligator Teeth (TEMA5) in opposite direction
# - Position sizing: 0.25 discrete level
# - Williams Alligator uses smoothed moving averages (SMMA) with specific periods
# - Target: 12-37 trades/year on 12h timeframe to stay within fee drag limits
# - Works in both bull/bear markets: Alligator identifies trend, volume confirms strength

name = "12h_1d_alligator_volume_v1"
timeframe = "12h"
leverage = 1.0

def _smma(arr, period):
    """Smoothed Moving Average (SMMA) - also called RMA or Wilder's MA"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=float)
    result = np.full_like(arr, np.nan, dtype=float)
    # First value is simple SMA
    result[period-1] = np.mean(arr[:period])
    # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current Price) / period
    for i in range(period, len(arr)):
        result[i] = (result[i-1] * (period-1) + arr[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 12h Williams Alligator components (TEMA-based approximation)
    # Jaw: TEMA13, Teeth: TEMA8, Lips: TEMA5
    close_series = pd.Series(close)
    ema5 = close_series.ewm(span=5, adjust=False, min_periods=5).mean()
    ema8 = close_series.ewm(span=8, adjust=False, min_periods=8).mean()
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean()
    
    # TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))
    ema5_2 = ema5.ewm(span=5, adjust=False, min_periods=5).mean()
    ema5_3 = ema5_2.ewm(span=5, adjust=False, min_periods=5).mean()
    tema5 = 3*ema5 - 3*ema5_2 + ema5_3
    
    ema8_2 = ema8.ewm(span=8, adjust=False, min_periods=8).mean()
    ema8_3 = ema8_2.ewm(span=8, adjust=False, min_periods=8).mean()
    tema8 = 3*ema8 - 3*ema8_2 + ema8_3
    
    ema13_2 = ema13.ewm(span=13, adjust=False, min_periods=13).mean()
    ema13_3 = ema13_2.ewm(span=13, adjust=False, min_periods=13).mean()
    tema13 = 3*ema13 - 3*ema13_2 + ema13_3
    
    jaw = tema13.values  # Alligator Jaw (blue line, 13-period)
    teeth = tema8.values   # Alligator Teeth (red line, 8-period)
    lips = tema5.values    # Alligator Lips (green line, 5-period)
    
    # Calculate 1d volume regime filter
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    for i in range(50, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(volume_1d_aligned[i]) or np.isnan(volume_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 2.0x 20-period volume SMA (strong participation)
        vol_confirm = volume_1d_aligned[i] > 2.0 * volume_sma_20_1d_aligned[i]
        
        # Alligator signals
        price_above_jaw = close[i] > jaw[i]
        price_below_lips = close[i] < lips[i]
        price_above_teeth = close[i] > teeth[i]
        price_below_teeth = close[i] < teeth[i]
        
        # Entry conditions: price outside lips/jaw with volume confirmation
        if position == 0:  # Flat - look for entry
            if price_above_jaw and vol_confirm:
                position = 1
                signals[i] = 0.25
            elif price_below_lips and vol_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            # Exit when price crosses teeth downward (trend weakening)
            if price_below_teeth:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            # Exit when price crosses teeth upward (trend weakening)
            if price_above_teeth:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals