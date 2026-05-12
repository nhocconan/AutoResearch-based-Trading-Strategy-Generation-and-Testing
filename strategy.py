#!/usr/bin/env python3
"""
1h_Combined_Momentum_Reversal
Hypothesis: Combines momentum (RSI) and mean-reversion (Bollinger Bands) on 1h timeframe,
filtered by 4h trend (EMA50) and 1d volume regime to avoid false signals. Uses volume spike
after low volatility to capture momentum bursts in both bull and bear markets. Designed for
low trade frequency (15-30/year) with discrete sizing (0.20) to minimize fee drag.
"""

name = "1h_Combined_Momentum_Reversal"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 4h data for trend filter (call once before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)

    # Calculate 4h EMA50 for trend filter
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1d data for volume regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume average and standard deviation for volatility regime
    vol_avg_20_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_std_20_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).std().values
    
    # Calculate RSI (14) on 1h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Bollinger Bands (20, 2) on 1h close
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20
    
    # Calculate volume spike detector (20-period average)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 1h timeframe
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    vol_std_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_std_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start from 50 to have enough data for all indicators
        # Get aligned values for current 1h bar
        ema50_val = ema50_4h_aligned[i]
        vol_avg_20_1d_val = vol_avg_20_1d_aligned[i]
        vol_std_20_1d_val = vol_std_20_1d_aligned[i]
        rsi_val = rsi[i]
        close_val = close[i]
        lower_band_val = lower_band[i]
        upper_band_val = upper_band[i]
        vol_avg_val = vol_avg_20[i]
        
        # Skip if any required data is NaN
        if (np.isnan(ema50_val) or np.isnan(vol_avg_20_1d_val) or np.isnan(vol_std_20_1d_val) or 
            np.isnan(rsi_val) or np.isnan(lower_band_val) or np.isnan(upper_band_val) or 
            np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Low volatility regime: 1d volume volatility < 50% of average
        low_vol_regime = vol_std_20_1d_val < (vol_avg_20_1d_val * 0.5)
        
        if position == 0:
            # LONG: RSI < 30 (oversold) + price at/below lower BB + 4h uptrend + low vol + volume spike
            if (rsi_val < 30 and 
                close_val <= lower_band_val and 
                close_val > ema50_val and 
                low_vol_regime and 
                volume[i] > vol_avg_val * 2.0):
                signals[i] = 0.20
                position = 1
            # SHORT: RSI > 70 (overbought) + price at/above upper BB + 4h downtrend + low vol + volume spike
            elif (rsi_val > 70 and 
                  close_val >= upper_band_val and 
                  close_val < ema50_val and 
                  low_vol_regime and 
                  volume[i] > vol_avg_val * 2.0):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI > 50 or price > upper BB or trend turns down
            if (rsi_val > 50 or close_val > upper_band_val or close_val < ema50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: RSI < 50 or price < lower BB or trend turns up
            if (rsi_val < 50 or close_val < lower_band_val or close_val > ema50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals