#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Keltner breakout with 4h EMA25 trend filter and volume confirmation.
# In 4h uptrend (price > EMA25), break above upper Keltner band (EMA + 2*ATR) signals momentum.
# In 4h downtrend (price < EMA25), break below lower Keltner band (EMA - 2*ATR) signals momentum.
# Volume > 1.5x average confirms institutional participation. Designed for 15-30 trades/year.
name = "1h_Keltner_Breakout_4hEMA25_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA25 and ATR
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 25:
        return np.zeros(n)
    
    # Calculate 25-period EMA on 4h close
    close_4h = df_4h['close'].values
    ema_25_4h = pd.Series(close_4h).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_25_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_25_4h)
    
    # Calculate ATR(14) on 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_4h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # Calculate 1h EMA for Keltner channels (EMA20)
    close_1h = close
    ema_20_1h = pd.Series(close_1h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate ATR(10) for 1h Keltner width
    tr1h = high - low
    tr2h = np.abs(high - np.roll(close, 1))
    tr3h = np.abs(low - np.roll(close, 1))
    tr1h[0] = tr2h[0] = tr3h[0] = np.nan
    trh = np.maximum(tr1h, np.maximum(tr2h, tr3h))
    atr_10_1h = pd.Series(trh).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner bands: EMA20 ± 2 * ATR10
    keltner_upper = ema_20_1h + 2.0 * atr_10_1h
    keltner_lower = ema_20_1h - 2.0 * atr_10_1h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(25, 20, 14, 10, 1)  # Need data for all indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(ema_25_4h_aligned[i]) or np.isnan(atr_14_4h_aligned[i]) or \
           np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_4h = ema_25_4h_aligned[i]
        atr_4h = atr_14_4h_aligned[i]
        keltner_up = keltner_upper[i]
        keltner_low = keltner_lower[i]
        vol = volume[i]
        
        # Calculate 20-period volume average for confirmation
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
        else:
            vol_ma = np.mean(volume[:i]) if i > 0 else volume[i]
        
        if position == 0:
            # Enter long: 4h uptrend AND price > upper Keltner AND volume > 1.5x average
            if close[i] > ema_4h and close[i] > keltner_up and vol > 1.5 * vol_ma:
                signals[i] = 0.20
                position = 1
            # Enter short: 4h downtrend AND price < lower Keltner AND volume > 1.5x average
            elif close[i] < ema_4h and close[i] < keltner_low and vol > 1.5 * vol_ma:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Price < lower Keltner OR 4h trend turns down
            if close[i] < keltner_low or close[i] < ema_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Price > upper Keltner OR 4h trend turns up
            if close[i] > keltner_up or close[i] > ema_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals