# Solution
#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 1-day Keltner channel breakout with 1-week EMA trend filter and volume confirmation.
In uptrend (price > 1w EMA100), buy breakouts above 1d Keltner upper channel; in downtrend (price < 1w EMA100), sell breakdowns below 1d Keltner lower channel.
Volume must exceed 1.5x 50-period average to confirm breakout strength. Exit on trend reversal or 2x ATR stop.
Designed for 20-50 trades/year (80-200 total over 4 years) to minimize fee decay while capturing major trend moves.
Works in bull markets via upper channel breakouts and in bear markets via lower channel breakdowns with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1-day data ONCE before loop for Keltner channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day Keltner channels (20-period EMA + 2*ATR)
    ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    keltner_upper = ema_20 + 2 * atr_1d
    keltner_lower = ema_20 - 2 * atr_1d
    
    # Load 1-week data ONCE before loop for EMA100 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 100:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_100_1w = pd.Series(close_1w).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Align 1d and 1w indicators to 4h timeframe (wait for bar to close)
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1d, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1d, keltner_lower)
    ema_100_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_100_1w)
    
    # Volume confirmation (volume spike > 1.5x 50-period average)
    vol_ma_50 = pd.Series(prices['volume'].values).rolling(window=50, min_periods=50).mean().values
    vol_ratio = prices['volume'].values / vol_ma_50
    
    # ATR for stoploss (20-period on 4h)
    tr1 = prices['high'].values - prices['low'].values
    tr2 = np.abs(prices['high'].values - np.roll(prices['close'].values, 1))
    tr3 = np.abs(prices['low'].values - np.roll(prices['close'].values, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(keltner_upper_aligned[i]) or np.isnan(keltner_lower_aligned[i]) or 
            np.isnan(ema_100_1w_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        keltner_up = keltner_upper_aligned[i]
        keltner_low = keltner_lower_aligned[i]
        ema_trend = ema_100_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        if position == 0:
            # Enter long: price breaks above 1d Keltner upper + uptrend + volume spike
            if (price_close > keltner_up and 
                price_close > ema_trend and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 1d Keltner lower + downtrend + volume spike
            elif (price_close < keltner_low and 
                  price_close < ema_trend and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: trend reversal OR ATR-based stoploss
            exit_signal = False
            
            # Trend reversal exit
            if position == 1 and price_close < ema_trend:
                exit_signal = True
            elif position == -1 and price_close > ema_trend:
                exit_signal = True
            
            # ATR-based stoploss (2x ATR from entry)
            if position == 1:
                # Approximate entry price as the Keltner upper breakout level
                entry_approx = keltner_upper_aligned[i-1] if i > 0 else keltner_upper_aligned[i]
                if price_close < entry_approx - 2.0 * atr_val:
                    exit_signal = True
            elif position == -1:
                # Approximate entry price as the Keltner lower breakdown level
                entry_approx = keltner_lower_aligned[i-1] if i > 0 else keltner_lower_aligned[i]
                if price_close > entry_approx + 2.0 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Keltner20_1wEMA100_Volume_ATR"
timeframe = "4h"
leverage = 1.0