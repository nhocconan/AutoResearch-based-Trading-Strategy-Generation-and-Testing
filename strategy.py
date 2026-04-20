#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Commodity Channel Index (CCI) with 1d EMA200 trend filter and volume confirmation.
# CCI identifies cyclical turning points and overbought/oversold conditions.
# Buy when CCI crosses above -100 from below (end of pullback) in uptrend (price > EMA200).
# Sell when CCI crosses below +100 from above (end of rally) in downtrend (price < EMA200).
# Volume confirmation ensures institutional participation.
# Works in both bull/bear by following higher timeframe trend and avoiding counter-trend trades.
# Target: 20-40 trades per year to minimize fee drag.

name = "4h_CCI_1dEMA200_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for EMA200 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d EMA200 for trend direction ===
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # === CCI(20) on 4h ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    typical_price = (high + low + close) / 3
    sma_tp = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    
    # Avoid division by zero
    cci = (typical_price - sma_tp) / (0.015 * np.where(mad > 0, mad, np.nan))
    
    # === 4h Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):  # Start after CCI warmup
        # Get values
        close_val = prices['close'].iloc[i]
        ema_val = ema_200_aligned[i]
        cci_val = cci[i]
        vol_ratio_val = vol_ratio[i]
        prev_cci = cci[i-1] if i > 0 else 0
        
        # Skip if any value is NaN
        if (np.isnan(ema_val) or np.isnan(cci_val) or np.isnan(vol_ratio_val) or 
            np.isnan(prev_cci)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: CCI crosses above -100 from below in uptrend (price > EMA200) with volume
            if (prev_cci <= -100 and cci_val > -100 and 
                close_val > ema_val and vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: CCI crosses below +100 from above in downtrend (price < EMA200) with volume
            elif (prev_cci >= 100 and cci_val < 100 and 
                  close_val < ema_val and vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        
        elif position == 1:
            # Long exit: CCI crosses below +100 (end of rally) or trend reversal
            if cci_val < 100 or close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: CCI crosses above -100 (end of pullback) or trend reversal
            if cci_val > -100 or close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals