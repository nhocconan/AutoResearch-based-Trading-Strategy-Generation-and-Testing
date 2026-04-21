#!/usr/bin/env python3
"""
1d_Retracement_to_MA_with_Volume_Spike
Hypothesis: In strong weekly trends, price retraces to weekly EMA21/50, offering high-probability entries. Combines weekly EMA trend filter with daily price retracement to that EMA and volume spike confirmation. Works in bull/bear markets by following the weekly trend. Targets 15-25 trades/year on daily timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === Weekly trend: EMA21 and EMA50 ===
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Daily price action: retracement to weekly EMA ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Distance from weekly EMAs (as % of ATR-like measure)
    # Use daily ATR(14) for normalization
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # === Volume confirmation: 20-period volume spike ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume / vol_ma_20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(ema_21_1w_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        price_low = low[i]
        price_high = high[i]
        ema21 = ema_21_1w_aligned[i]
        ema50 = ema_50_1w_aligned[i]
        atr_val = atr[i]
        vol_spike = vol_ratio[i]
        
        # Normalized distance to EMAs
        dist_to_ema21 = abs(price_close - ema21) / atr_val if atr_val > 0 else 0
        dist_to_ema50 = abs(price_close - ema50) / atr_val if atr_val > 0 else 0
        
        if position == 0:
            # Long: Weekly uptrend (EMA21 > EMA50) + price retraces to EMA21/50 + volume spike
            if (ema21 > ema50 and 
                dist_to_ema21 < 0.5 and  # Within 0.5 ATR of EMA21
                vol_spike > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Weekly downtrend (EMA21 < EMA50) + price retraces to EMA21/50 + volume spike
            elif (ema21 < ema50 and 
                  dist_to_ema21 < 0.5 and  # Within 0.5 ATR of EMA21
                  vol_spike > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price moves 1.5 ATR away from EMA in opposite direction OR trend reverses
            if position == 1:
                if (price_close > ema21 + 1.5 * atr_val) or (ema21 < ema50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if (price_close < ema21 - 1.5 * atr_val) or (ema21 > ema50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Retracement_to_MA_with_Volume_Spike"
timeframe = "1d"
leverage = 1.0