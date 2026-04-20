#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot reversal with daily volume confirmation and 1d ATR stop
# - Uses daily Camarilla pivot levels (R1, S1) as reversal zones
# - Long: price crosses below S1 with volume > 1.5x daily avg, expecting bounce
# - Short: price crosses above R1 with volume > 1.5x daily avg, expecting rejection
# - Trend filter: price must be below 1d EMA50 for longs, above for shorts
# - Exit: price crosses back to pivot point (PP) or 1d ATR stop (1.5x ATR)
# - Designed for low-frequency, high-conviction reversals in ranging markets
# - Target: 15-25 trades per year per symbol (60-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for Camarilla pivots and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (using previous day's OHLC)
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pp_1d = (high_1d + low_1d + close_1d) / 3
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d ATR for stop loss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily volume average for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all daily data to 12h timeframe
    pp_12h = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    atr_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    vol_ma_12h = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 12h price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if (np.isnan(pp_12h[i]) or np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or 
            np.isnan(ema_50_12h[i]) or np.isnan(atr_12h[i]) or np.isnan(vol_ma_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long setup: price below S1 (oversold) with volume surge, expecting bounce
            # Only take if below EMA50 (downtrend) for mean reversion
            if price < s1_12h[i] and price > s1_12h[i-1] and vol > 1.5 * vol_ma_12h[i] and price < ema_50_12h[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short setup: price above R1 (overbought) with volume surge, expecting rejection
            # Only take if above EMA50 (uptrend) for mean reversion
            elif price > r1_12h[i] and price < r1_12h[i-1] and vol > 1.5 * vol_ma_12h[i] and price > ema_50_12h[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price crosses back to PP (mean reversion complete) OR ATR stop
            if price > pp_12h[i] or price < entry_price - 1.5 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back to PP (mean reversion complete) OR ATR stop
            if price < pp_12h[i] or price > entry_price + 1.5 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_PP_R1S1_Volume_EMA50Filter"
timeframe = "12h"
leverage = 1.0