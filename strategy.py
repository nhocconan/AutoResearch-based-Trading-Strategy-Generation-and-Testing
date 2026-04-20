#!/usr/bin/env python3
# 1d_1w_MomentumReversal_Volume_Confirm
# Hypothesis: On daily timeframe, buy when RSI(14) < 30 (oversold) with volume confirmation and weekly trend filter (price > weekly EMA50).
# Sell when RSI(14) > 70 (overbought) with volume confirmation and weekly trend filter (price < weekly EMA50).
# Uses weekly EMA50 to filter counter-trend trades in strong trends, reducing whipsaw in bear markets.
# Target: 15-25 trades per year per symbol to avoid fee drag, works in bull/bear via trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_MomentumReversal_Volume_Confirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === Weekly EMA50 for trend filter ===
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === Daily RSI(14) ===
    close = prices['close'].values
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # === Daily volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after RSI and volume MA warmup
        # Get values
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio[i]
        ema50_1w_val = ema50_1w_aligned[i]
        close_val = close[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi_val) or np.isnan(vol_ratio_val) or np.isnan(ema50_1w_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) with volume confirmation and above weekly EMA50
            if (rsi_val < 30 and vol_ratio_val > 2.0 and close_val > ema50_1w_val):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) with volume confirmation and below weekly EMA50
            elif (rsi_val > 70 and vol_ratio_val > 2.0 and close_val < ema50_1w_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral (>= 50) or weekly trend fails
            if rsi_val >= 50 or close_val < ema50_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral (<= 50) or weekly trend fails
            if rsi_val <= 50 or close_val > ema50_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals