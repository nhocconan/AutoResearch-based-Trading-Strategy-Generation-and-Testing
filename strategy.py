#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA trend with 1d ATR regime filter and volume spike
# - KAMA adapts to market efficiency, reducing whipsaw in ranging markets
# - 1d ATR ratio (current/20-day average) > 1.5 identifies high-volatility regimes
# - 1d volume spike (>1.8x 20-day average) confirms institutional participation
# - Discrete position sizing (0.25) minimizes fee churn
# - Target: 30-50 trades/year (120-200 total over 4 years) to balance edge and fees
# - Works in bull markets via trend following, in bear markets via volatility filters that reduce false signals

name = "4h_1d_kama_atr_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d ATR and its moving average
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_20_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20_1d)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Pre-compute 1d volume and its moving average
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    volume_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    # Pre-compute 4h KAMA (adaptive moving average)
    close_4h = prices['close'].values
    direction = np.abs(np.diff(close_4h, 10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close_4h, 1)), axis=1)  # 10-period sum of abs changes
    er = np.where(volatility > 0, direction / volatility, 0)
    # Pad ER array to match length
    er = np.concatenate([np.full(9, np.nan), er])
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # smoothing constant
    kama = np.full_like(close_4h, np.nan)
    kama[9] = close_4h[9]  # seed
    for i in range(10, len(close_4h)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_4h[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(atr_aligned[i]) or 
            np.isnan(atr_ma_aligned[i]) or np.isnan(volume_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Regime filter: 1d ATR ratio > 1.5 (high volatility regime)
        atr_ratio = atr_aligned[i] / atr_ma_aligned[i]
        high_vol_regime = atr_ratio > 1.5
        
        # Volume confirmation: current 1d volume > 1.8x 20-day average
        volume_confirm = volume_aligned[i] > 1.8 * volume_ma_aligned[i]
        
        close_price = close_4h[i]
        kama_value = kama[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price > KAMA AND high volatility regime AND volume confirmation
            if close_price > kama_value and high_vol_regime and volume_confirm:
                position = 1
                signals[i] = 0.25
            # Short conditions: price < KAMA AND high volatility regime AND volume confirmation
            elif close_price < kama_value and high_vol_regime and volume_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price crosses KAMA in opposite direction
            if position == 1 and close_price < kama_value:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close_price > kama_value:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals