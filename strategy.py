#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h momentum strategy using RSI and volume divergence to catch trend reversals.
# Uses RSI(14) for momentum, volume confirmation for institutional participation, 
# and 1d EMA200 for higher-timeframe trend filter to avoid counter-trend trades.
# Strategy goes long when RSI crosses above 50 (bullish momentum) with volume > 1.5x average
# and price above 1d EMA200. Short when RSI crosses below 50 with volume confirmation and price below 1d EMA200.
# This captures momentum shifts while filtering for trend alignment and institutional interest.
# Target: 20-40 trades per year to minimize fee drag.

name = "4h_RSI_Momentum_1dEMA200_Volume"
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
    
    # === RSI(14) for momentum ===
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (equivalent to EMA with alpha=1/period)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(14, n):  # Start after RSI warmup
        # Get values
        close_val = prices['close'].iloc[i]
        ema_val = ema_200_aligned[i]
        rsi_val = rsi[i]
        rsi_prev = rsi[i-1]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_val) or np.isnan(rsi_val) or np.isnan(rsi_prev) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: RSI crosses above 50 (bullish momentum) with volume confirmation
            # and price above 1d EMA200 (trend alignment)
            if rsi_prev <= 50 and rsi_val > 50 and close_val > ema_val and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short entry: RSI crosses below 50 (bearish momentum) with volume confirmation
            # and price below 1d EMA200 (trend alignment)
            elif rsi_prev >= 50 and rsi_val < 50 and close_val < ema_val and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        
        elif position == 1:
            # Long exit: RSI crosses below 50 (loss of bullish momentum) 
            # or price closes below 1d EMA200 (trend reversal)
            if rsi_val < 50 or close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI crosses above 50 (loss of bearish momentum)
            # or price closes above 1d EMA200 (trend reversal)
            if rsi_val > 50 or close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals