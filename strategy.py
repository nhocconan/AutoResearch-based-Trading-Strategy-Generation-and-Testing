#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h ATR-based breakout with 12h trend filter and volume confirmation.
# Uses ATR(14) to measure volatility and breakout from ATR-based channels.
# Trend filter uses 12h EMA50 to align with higher timeframe direction.
# Volume confirmation ensures institutional participation.
# Designed to work in both bull and bear markets by following higher timeframe trend
# and avoiding false breakouts in low-volume conditions.
# Target: 20-40 trades per year to minimize fee drag.

name = "4h_ATRBreakout_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop for EMA50 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # === 12h EMA50 for trend direction ===
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # === ATR(14) calculation ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range components
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # ATR using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14  # Wilder's smoothing
    
    # === ATR-based channels ===
    atr_mult = 1.5
    upper_channel = close + atr_mult * atr
    lower_channel = close - atr_mult * atr
    
    # === Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Get values
        close_val = prices['close'].iloc[i]
        ema_val = ema_50_aligned[i]
        upper_val = upper_channel[i]
        lower_val = lower_channel[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_val) or np.isnan(upper_val) or np.isnan(lower_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price closes above upper channel with uptrend and volume
            if close_val > upper_val and ema_val > close_val and vol_ratio_val > 1.8:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short breakout: price closes below lower channel with downtrend and volume
            elif close_val < lower_val and ema_val < close_val and vol_ratio_val > 1.8:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        
        elif position == 1:
            # Long exit: price closes below lower channel or trend reversal
            if close_val < lower_val or ema_val < close_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above upper channel or trend reversal
            if close_val > upper_val or ema_val > close_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals