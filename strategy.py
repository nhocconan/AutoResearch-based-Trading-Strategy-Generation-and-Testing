#!/usr/bin/env python3
# 1d_1w_keltner_trend_reversion_v1
# Strategy: 1-day Keltner Channel mean reversion with 1-week trend filter
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: In crypto markets, price often reverts to the mean after deviating
# significantly from the Keltner Channel (2x ATR) during overbought/oversold
# conditions. The 1-week EMA(20) trend filter ensures we only take mean-reversion
# trades in the direction of the higher timeframe trend, improving win rate.
# Works in bull markets by buying dips in uptrends and in bear markets by selling
# rallies in downtrends. Low trade frequency (target: 15-30/year) minimizes fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_keltner_trend_reversion_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily ATR(14) for Keltner Channel
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Keltner Channel (20-period EMA ± 2*ATR)
    keltner_mid = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    keltner_upper = keltner_mid + 2.0 * atr
    keltner_lower = keltner_mid - 2.0 * atr
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Keltner warmup
        # Skip if any required data is invalid
        if (np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or 
            np.isnan(keltner_mid[i]) or np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Mean reversion conditions
        touch_upper = close[i] >= keltner_upper[i]   # Touch or break upper band
        touch_lower = close[i] <= keltner_lower[i]   # Touch or break lower band
        
        # Trend filter: price relative to weekly EMA20
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Entry logic: mean reversion with trend alignment
        if touch_lower and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif touch_upper and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: price returns to middle of Keltner Channel
        elif position == 1 and close[i] <= keltner_mid[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= keltner_mid[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals