#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Donchian breakout captures momentum; EMA50 ensures alignment with higher timeframe trend
# Volume confirmation filters false breakouts; targets 20-50 trades/year for low friction
# Works in bull (breakouts up) and bear (breakouts down) via symmetric long/short logic
# Uses discrete position sizing (0.25) to minimize churn; includes ATR-based stoploss

name = "4h_Donchian20_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # 1d EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(lookback, 50)  # Ensure sufficient data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        highest_high_val = highest_high[i]
        lowest_low_val = lowest_low[i]
        ema50_1d_val = ema50_1d_aligned[i]
        vol_conf_val = vol_conf[i]
        atr_val = atr[i]
        
        if position == 0:
            # Enter long: breakout above Donchian high, uptrend on 1d, volume confirmation
            if close_val > highest_high_val and ema50_1d_val > 0 and vol_conf_val:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Enter short: breakout below Donchian low, downtrend on 1d, volume confirmation
            elif close_val < lowest_low_val and ema50_1d_val < 0 and vol_conf_val:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long exit: stoploss, trend reversal, or opposite breakout
            if (close_val <= entry_price - 2.0 * atr_val or  # Stoploss
                ema50_1d_val < 0 or                          # Trend reversal
                close_val < lowest_low_val):                 # Opposite signal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: stoploss, trend reversal, or opposite breakout
            if (close_val >= entry_price + 2.0 * atr_val or  # Stoploss
                ema50_1d_val > 0 or                          # Trend reversal
                close_val > highest_high_val):               # Opposite signal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals