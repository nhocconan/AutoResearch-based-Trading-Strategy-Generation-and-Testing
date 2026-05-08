#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Keltner Channel breakout with 1-day ATR volatility filter and volume confirmation
# Long when price > upper KC (EMA20 + 2*ATR10) and daily ATR10 > daily ATR30 (vol expansion)
# Short when price < lower KC (EMA20 - 2*ATR10) and daily ATR10 > daily ATR30 (vol expansion)
# Keltner Channel adapts to volatility better than fixed bands; volatility expansion filters breakouts
# Volume confirmation ensures institutional participation; avoids low-volume false breakouts
# Targets 50-150 total trades over 4 years (12-37/year) on 12h timeframe to minimize fee drag

name = "12h_Keltner_VolATR_Filter_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data once for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ATR(10) and ATR(30) for volatility filter
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # True Range calculation
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.roll(daily_close, 1))
    tr3 = np.abs(daily_low - np.roll(daily_close, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr10_1d = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr30_1d = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    atr10_1d_aligned = align_htf_to_ltf(prices, df_1d, atr10_1d)
    atr30_1d_aligned = align_htf_to_ltf(prices, df_1d, atr30_1d)
    
    # Calculate EMA(20) for Keltner Channel
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate ATR(10) for Keltner Channel width
    tr_l = high - low
    tr_h = np.abs(high - np.roll(close, 1))
    tr_lw = np.abs(low - np.roll(close, 1))
    tr_l[0] = 0
    tr_h[0] = 0
    tr_lw[0] = 0
    tr_lcl = np.maximum(tr_l, np.maximum(tr_h, tr_lw))
    atr10 = pd.Series(tr_lcl).rolling(window=10, min_periods=10).mean().values
    
    # Keltner Channel bands
    kc_upper = ema20 + 2.0 * atr10
    kc_lower = ema20 - 2.0 * atr10
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema20[i]) or np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or 
            np.isnan(atr10_1d_aligned[i]) or np.isnan(atr30_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        kc_up = kc_upper[i]
        kc_low = kc_lower[i]
        atr10_1d_val = atr10_1d_aligned[i]
        atr30_1d_val = atr30_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price > upper KC and daily vol expansion and volume spike
            if price > kc_up and atr10_1d_val > atr30_1d_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: price < lower KC and daily vol expansion and volume spike
            elif price < kc_low and atr10_1d_val > atr30_1d_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < EMA20 or volatility contraction
            if price < ema20[i] or atr10_1d_val <= atr30_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > EMA20 or volatility contraction
            if price > ema20[i] or atr10_1d_val <= atr30_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals