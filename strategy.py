# 100855 - 6h_Keltner_Channel_Retest_with_RSI_Filter
# Hypothesis: Price retests Keltner Channel bands (EMA20 ± 2*ATR10) with RSI(14) showing momentum divergence.
# Works in bull (retest support in uptrend) and bear (retest resistance in downtrend).
# Uses 6h primary with 1d EMA50 trend filter. Targets 15-25 trades/year via strict retest conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Keltner Channel components
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr10 = pd.Series(np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))).rolling(window=10, min_periods=10).mean().values
    upper_keltner = ema20 + 2 * atr10
    lower_keltner = ema20 - 2 * atr10
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20[i]) or np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or 
            np.isnan(rsi[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long: retest lower band in uptrend with RSI > 50 (bullish momentum)
        if (low[i] <= lower_keltner[i] * 1.001 and  # Allow small tolerance for retest
            close[i] > ema50_1d_aligned[i] and      # Uptrend filter
            rsi[i] > 50):
            signals[i] = 0.25
            position = 1
        # Short: retest upper band in downtrend with RSI < 50 (bearish momentum)
        elif (high[i] >= upper_keltner[i] * 0.999 and  # Allow small tolerance for retest
              close[i] < ema50_1d_aligned[i] and      # Downtrend filter
              rsi[i] < 50):
            signals[i] = -0.25
            position = -1
        # Exit: price returns to EMA20 (mean reversion to middle)
        elif position == 1 and close[i] < ema20[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > ema20[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Keltner_Channel_Retest_with_RSI_Filter"
timeframe = "6h"
leverage = 1.0