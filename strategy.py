#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_kama_volatility_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return signals
    
    # Calculate KAMA on 12h close
    close_12h = df_12h['close'].values
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    volatility = np.sum(np.abs(np.diff(close_12h)), axis=0)
    # ER = |change| / volatility, avoid division by zero
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(close_12h)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    
    # Calculate ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align 12h KAMA to 4h
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    
    # Volatility filter: ATR > 1.5 * ATR(50)
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr > 1.5 * atr_50
    
    # Volume confirmation: volume > 2.0 * volume MA(20)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 2.0 * vol_ma_20
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(atr_50[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        kama_val = kama_aligned[i]
        vol_filt = volatility_filter[i]
        vol_conf = volume_confirmed[i]
        
        # Entry signals
        long_signal = price_close > kama_val and vol_filt and vol_conf
        short_signal = price_close < kama_val and vol_filt and vol_conf
        
        # Exit conditions
        exit_long = position == 1 and price_close < kama_val
        exit_short = position == -1 and price_close > kama_val
        # Stop loss
        stop_long = position == 1 and price_low < (entry_price - 2.5 * atr[i])
        stop_short = position == -1 and price_high > (entry_price + 2.5 * atr[i])
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            entry_price = price_close
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            entry_price = price_close
            signals[i] = -0.25
        elif position == 1 and (exit_long or stop_long):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (exit_short or stop_short):
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: KAMA volatility breakout strategy with volume confirmation on 4h timeframe.
# Uses 12h KAMA to filter noise and identify the true trend direction.
# Enters long when price crosses above 12h KAMA with high volatility (ATR > 1.5*ATR50) and high volume (>2x avg volume).
# Enters short when price crosses below 12h KAMA with same conditions.
# Uses ATR-based stop loss (2.5x) and exits when price crosses back below/above KAMA.
# Designed for low trade frequency (<50 trades/year) to minimize fee drag.
# Works in both bull and bear markets by adapting to volatility regimes and using KAMA's adaptive smoothing.