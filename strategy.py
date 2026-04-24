#!/usr/bin/env python3
"""
Hypothesis: 1d Williams Alligator + Elder Ray combo with 1w EMA50 trend filter, volume confirmation (>2.0x 24-bar average), and ATR regime filter (current ATR > 0.7x 50-bar average).
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 20-30 trades/year (80-120 total over 4 years) to stay fee-efficient.
- Combines Alligator trend identification + Elder Ray bull/bear power + 1w trend filter.
- Works in bull/bear: 1w EMA50 ensures alignment with weekly trend; volume/volatility filters avoid low-conviction entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior 1d OHLC (completed 1d bar)
    high_1d = df_1d['high'].shift(1).values
    low_1d = df_1d['low'].shift(1).values
    close_1d = df_1d['close'].shift(1).values
    
    # Align to 1d timeframe (primary timeframe is 1d, so no alignment needed for 1d->1d)
    # But we still use align_htf_to_ltf for safety and consistency
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Williams Alligator: SMAs of median price
    median_price = (high_1d_aligned + low_1d_aligned) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # 13-period
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values   # 8-period
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values    # 5-period
    
    # Shift Alligator lines by 3, 2, 1 bars respectively (as per Williams)
    jaw = np.roll(jaw, 3)
    teeth = np.roll(teeth, 2)
    lips = np.roll(lips, 1)
    # Set NaN for rolled values
    jaw[:3] = np.nan
    teeth[:2] = np.nan
    lips[:1] = np.nan
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close_1d_aligned).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d_aligned - ema_13
    bear_power = low_1d_aligned - ema_13
    
    # 1w EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].shift(1).values  # Prior completed 1w bar
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    ema_50_1w = pd.Series(close_1w_aligned).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: > 2.0x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # ATR(14) for volatility regime filter
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # ATR ratio: current ATR / 50-period average (avoid low volatility chop)
    atr_ma_long = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / np.where(atr_ma_long > 0, atr_ma_long, 1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 8, 5, 13, 24, 14, 50, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_50_1w[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average) + ATR ratio > 0.7 (avoid low vol)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        vol_regime = atr_ratio[i] > 0.7
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (Alligator bullish) AND Bull Power > 0 AND price above 1w EMA50 AND volume confirmation AND vol regime
            if (lips[i] > teeth[i] > jaw[i]) and (bull_power[i] > 0) and (close[i] > ema_50_1w[i]) and volume_confirm and vol_regime:
                signals[i] = 0.25
                position = 1
            # Short: Jaw > Teeth > Lips (Alligator bearish) AND Bear Power < 0 AND price below 1w EMA50 AND volume confirmation AND vol regime
            elif (jaw[i] > teeth[i] > lips[i]) and (bear_power[i] < 0) and (close[i] < ema_50_1w[i]) and volume_confirm and vol_regime:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator turns bearish (Jaw > Teeth > Lips) OR Bear Power < 0
            if (jaw[i] > teeth[i] > lips[i]) or (bear_power[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator turns bullish (Lips > Teeth > Jaw) OR Bull Power > 0
            if (lips[i] > teeth[i] > jaw[i]) or (bull_power[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsAlligator_ElderRay_1wEMA50_VolumeATR_Filter_v1"
timeframe = "1d"
leverage = 1.0