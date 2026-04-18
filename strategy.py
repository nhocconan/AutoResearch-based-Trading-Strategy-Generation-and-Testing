#!/usr/bin/env python3
"""
1d_Keltner_MeanReversion_with_Target
Hypothesis: Mean reversion from Keltner Channel extremes with RSI confirmation and profit target.
Uses 1d Keltner Channels (EMA20 ± 2*ATR10), RSI(14) for overbought/oversold, and fixed profit target.
Designed to work in both bull and bear markets by fading extremes during low volatility periods.
Target: 8-25 trades/year (32-100 total over 4 years) to minimize fee drift while capturing mean reversion moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly ATR for regime filter (low volatility = mean reversion favorable)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range and ATR(10) for weekly
    tr1_w = high_1w[1:] - low_1w[1:]
    tr2_w = np.abs(high_1w[1:] - close_1w[:-1])
    tr3_w = np.abs(low_1w[1:] - close_1w[:-1])
    tr_w = np.maximum(np.maximum(tr1_w, tr2_w), tr3_w)
    tr_w = np.concatenate([[np.nan], tr_w])
    atr_10_1w = pd.Series(tr_w).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr_10_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_10_1w)
    
    # Daily Keltner Channel: EMA20 ± 2*ATR10
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    upper_keltner = ema_20 + (2.0 * atr_10)
    lower_keltner = ema_20 - (2.0 * atr_10)
    
    # RSI(14) for overbought/oversold confirmation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume filter: above average to avoid dead cat bounces
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0
    entry_price = 0.0
    
    start_idx = max(20, 14)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20[i]) or np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or
            np.isnan(rsi_values[i]) or np.isnan(volume_filter[i]) or np.isnan(atr_10_1w_aligned[i])):
            signals[i] = 0.0
            if position != 0:
                # Maintain position until exit conditions
                signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        price = close[i]
        upper = upper_keltner[i]
        lower = lower_keltner[i]
        rsi_val = rsi_values[i]
        vol_filter = volume_filter[i]
        weekly_atr = atr_10_1w_aligned[i]
        
        # Skip if weekly volatility is too high (avoid mean reversion in strong trends)
        if weekly_atr > 0 and atr_10[i] > (1.5 * weekly_atr):
            if position != 0:
                signals[i] = 0.25 if position == 1 else -0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price at lower Keltner band with oversold RSI and volume
            if (price <= lower and
                rsi_val < 30 and
                vol_filter):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price at upper Keltner band with overbought RSI and volume
            elif (price >= upper and
                  rsi_val > 70 and
                  vol_filter):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: profit target (2.5% gain) or mean reversion failure (RSI > 50)
            if (price >= entry_price * 1.025 or  # 2.5% profit target
                rsi_val > 50):                   # RSI back to neutral
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: profit target (2.5% gain) or mean reversion failure (RSI < 50)
            if (price <= entry_price * 0.975 or  # 2.5% profit target
                rsi_val < 50):                   # RSI back to neutral
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Keltner_MeanReversion_with_Target"
timeframe = "1d"
leverage = 1.0