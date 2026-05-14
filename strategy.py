#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray Bull/Bear Power with 1d EMA34 trend filter and ATR volume spike confirmation.
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13.
# In strong uptrends (price > 1d EMA34), look for Bull Power expansion with volume spike to go long.
# In strong downtrends (price < 1d EMA34), look for Bear Power expansion with volume spike to go short.
# Uses discrete sizing (0.0, ±0.25) to minimize fee churn. Targets 12-25 trades/year per symbol.

name = "6h_ElderRay_BullBearPower_1dEMA34_ATRVolumeSpike_v1"
timeframe = "6h"
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
    
    # --- 6h Indicators (LTF) ---
    # EMA13 for Elder Ray calculation
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # ATR(14) for volatility normalization and volume spike
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
    volume_spike = vol_atr_ratio > (1.5 * vol_atr_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA34 on 1d for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(ema34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade in alignment with 1d EMA34
        if close[i] > ema34_1d_aligned[i]:
            # Uptrend: look for long signals
            if position == 0:
                # ENTER LONG: Bull Power expansion (> 0) AND volume spike
                if bull_power[i] > 0 and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif position == 1:
                # HOLD LONG: Continue if Bull Power still positive
                if bull_power[i] > 0:
                    signals[i] = 0.25
                else:
                    signals[i] = 0.0
                    position = 0
            elif position == -1:
                # EXIT SHORT: Close short if price crosses above EMA34
                signals[i] = 0.0
                position = 0
        else:
            # Downtrend: look for short signals
            if position == 0:
                # ENTER SHORT: Bear Power expansion (< 0) AND volume spike
                if bear_power[i] < 0 and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif position == -1:
                # HOLD SHORT: Continue if Bear Power still negative
                if bear_power[i] < 0:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
                    position = 0
            elif position == 1:
                # EXIT LONG: Close long if price crosses below EMA34
                signals[i] = 0.0
                position = 0
    
    return signals