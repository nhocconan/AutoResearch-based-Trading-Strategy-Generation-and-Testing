# 4h_Keltner_Channel_Pullback_With_Trend
# Hypothesis: In trending markets, price tends to pull back to the 20-period EMA before resuming the trend.
# We use 1-day EMA50 as the primary trend filter and enter on 4-hour pullbacks to EMA20 using Keltner channels.
# Long when: 1d trend up (close > EMA50_1d) AND price pulls back to touch lower Keltner band from below.
# Short when: 1d trend down (close < EMA50_1d) AND price pulls back to touch upper Keltner band from above.
# Keltner channels (EMA20 ± 2*ATR) provide dynamic support/resistance that adapts to volatility.
# Works in both bull (follows strong uptrends) and bear (follows strong downtrends).
# Uses volume confirmation to avoid low-conviction breakouts.

name = "4h_Keltner_Channel_Pullback_With_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA20 on 4h chart
    close_s = pd.Series(close)
    ema_20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for Keltner channels
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Keltner channels: EMA20 ± 2*ATR
    keltner_upper = ema_20 + 2 * atr
    keltner_lower = ema_20 - 2 * atr
    
    # Volume confirmation (20-period MA on 4h chart = ~3.3 days)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA20 (20), EMA50_1d (50), ATR (14), volume MA (20)
    start_idx = max(20, 50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_20[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(keltner_upper[i]) or 
            np.isnan(keltner_lower[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Price relative to Keltner channels for pullback detection
        if i > 0:
            touch_lower_from_below = (low[i] <= keltner_lower[i]) and (low[i-1] > keltner_lower[i-1])
            touch_upper_from_above = (high[i] >= keltner_upper[i]) and (high[i-1] < keltner_upper[i-1])
        else:
            touch_lower_from_below = False
            touch_upper_from_above = False
        
        if position == 0:
            # Long entry: uptrend + touch lower Keltner from below + volume
            if uptrend and touch_lower_from_below and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + touch upper Keltner from above + volume
            elif downtrend and touch_upper_from_above and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or reversal signal (touch upper band)
            if not uptrend or touch_upper_from_above:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or reversal signal (touch lower band)
            if not downtrend or touch_lower_from_below:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals