#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) with 1d EMA34 trend filter and ATR-based volume spike.
# Uses Alligator crossover for entry timing, 1d EMA34 for primary trend direction, and volume spike for conviction.
# Designed to catch trending moves with low frequency (target 12-30 trades/year) to minimize fee drag.
# Works in bull/bear via trend filter and mean-reversion exits on Alligator re-cross.

name = "12h_WilliamsAlligator_1dEMA34_ATRVolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 12h Indicators (LTF) ---
    # Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) SMAs of median price
    median_price = (high + low) / 2.0
    
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # ATR(14) for volatility and volume normalization
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    close_shift = np.roll(close, 1)
    high_shift[0] = high[0]
    low_shift[0] = low[0]
    close_shift[0] = close[0]
    
    tr = np.maximum(high - low, np.maximum(np.abs(high - close_shift), np.abs(low - close_shift)))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR-scaled volume MA: 20-period average of volume / ATR
    vol_atr_ratio = volume / (atr_14 + 1e-10)
    vol_atr_ma_20 = pd.Series(vol_atr_ratio).rolling(window=20, min_periods=20).mean().values
    volume_spike = vol_atr_ratio > (2.0 * vol_atr_ma_20)  # stricter spike for lower frequency
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # EMA34 on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade when price is above/below 1d EMA34
        if close[i] > ema34_1d_aligned[i]:
            trend = 1  # bullish bias
        elif close[i] < ema34_1d_aligned[i]:
            trend = -1  # bearish bias
        else:
            trend = 0
        
        if position == 0:
            # Wait for Alligator alignment with trend and volume spike
            if trend == 1 and lips[i] > teeth[i] > jaw[i] and volume_spike[i]:
                # Bullish alignment: Lips > Teeth > Jaw
                signals[i] = 0.25
                position = 1
            elif trend == -1 and lips[i] < teeth[i] < jaw[i] and volume_spike[i]:
                # Bearish alignment: Lips < Teeth < Jaw
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator re-cross (Teeth crosses below Jaw) or loss of bullish alignment
            if teeth[i] < jaw[i] or lips[i] < teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator re-cross (Teeth crosses above Jaw) or loss of bearish alignment
            if teeth[i] > jaw[i] or lips[i] > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals