# Your solution code here
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Keltner Channel breakouts with trend filter and volume confirmation
# Long when price breaks above upper KC with price > 200-bar EMA and volume > 1.5x average
# Short when price breaks below lower KC with price < 200-bar EMA and volume > 1.5x average
# Uses 1-day Keltner Channel for dynamic volatility-based support/resistance, EMA200 for trend filter, volume for confirmation
# Target: 25-40 trades per year (100-160 over 4 years) with 0.25 position sizing
# Works in bull markets via breakouts above resistance and in bear markets via breakdowns below support

name = "4h_1dKeltner_UpperLower_EMA200_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA200 on 4h close (needs 200 bars)
    close_series = pd.Series(close)
    ema200 = close_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Calculate 1-day Keltner Channel
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high, low, close
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    prev_close = df_1d['close'].shift(1)
    
    # True Range for ATR calculation
    tr1 = prev_high - prev_low
    tr2 = np.abs(prev_high - prev_close)
    tr3 = np.abs(prev_low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(10) for Keltner Channel width
    atr = pd.Series(tr).ewm(span=10, min_periods=10, adjust=False).mean().values
    
    # Keltner Channel: EMA(20) ± 2*ATR
    ema20 = pd.Series(prev_close).ewm(span=20, min_periods=20, adjust=False).mean().values
    kc_upper = ema20 + (2 * atr)
    kc_lower = ema20 - (2 * atr)
    
    # Align Keltner Channel levels to 4h timeframe
    kc_upper_aligned = align_htf_to_ltf(prices, df_1d, kc_upper)
    kc_lower_aligned = align_htf_to_ltf(prices, df_1d, kc_lower)
    
    # Volume confirmation: >1.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (1.5 * vol_ma_50)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(kc_upper_aligned[i]) or np.isnan(kc_lower_aligned[i]) or 
            np.isnan(ema200[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above upper KC with uptrend and volume confirmation
            if close[i] > kc_upper_aligned[i] and close[i] > ema200[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below lower KC with downtrend and volume confirmation
            elif close[i] < kc_lower_aligned[i] and close[i] < ema200[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower KC (support break)
            if close[i] < kc_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper KC (resistance break)
            if close[i] > kc_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals