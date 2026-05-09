#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Keltner Channel breakout with 1-week EMA50 trend filter and volume spike confirmation.
# Keltner Channel (ATR-based) captures volatility breakouts, weekly EMA50 ensures trend alignment,
# volume spike (>1.5x average) confirms institutional interest. Designed to work in both bull
# and bear markets by following the weekly trend direction. Target: 20-80 total trades over 4 years.
name = "1d_KeltnerBreakout_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-week data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 1w close
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(20) for Keltner Channel
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Keltner Channel: EMA(20) ± 2*ATR
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    kc_upper = ema_20 + 2.0 * atr
    kc_lower = ema_20 - 2.0 * atr
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need 50 periods for 1w EMA50 and ATR20
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or np.isnan(ema_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1w = ema_50_1w_aligned[i]
        upper = kc_upper[i]
        lower = kc_lower[i]
        ema20 = ema_20[i]
        vol = volume[i]
        
        # Calculate 20-period volume average for spike detection
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
        else:
            vol_ma = np.mean(volume[:i]) if i > 0 else volume[i]
        
        if position == 0:
            # Enter long: Close > KC Upper AND price > 1w EMA50 (uptrend) AND volume > 1.5x average
            if close[i] > upper and close[i] > ema_1w and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Enter short: Close < KC Lower AND price < 1w EMA50 (downtrend) AND volume > 1.5x average
            elif close[i] < lower and close[i] < ema_1w and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close < EMA(20) OR trend reverses (price < 1w EMA50)
            if close[i] < ema20 or close[i] < ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close > EMA(20) OR trend reverses (price > 1w EMA50)
            if close[i] > ema20 or close[i] > ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals