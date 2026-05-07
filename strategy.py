# 1d_RSI2_Trend_Filter_Volume_Spike
# Hypothesis: Uses 1d RSI(2) with trend filter (1w EMA50) and volume confirmation for high-probability entries.
# RSI(2) identifies short-term extremes while the 1w EMA50 ensures trend alignment.
# Volume spike confirms institutional participation. Designed for low trade frequency (<25/year) to minimize fee drag.
# Works in bull/bear markets by following the weekly trend direction.

timeframe = "1d"
name = "1d_RSI2_Trend_Filter_Volume_Spike"
leverage = 1.0

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
    
    # Get weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily RSI(2) for entry signals
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_2 = 100 - (100 / (1 + rs))
    rsi_2_values = rsi_2.fillna(50).values  # Neutral RSI when insufficient data
    
    # Volume spike detection: 2x average volume (20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure we have EMA50 and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(rsi_2_values[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI(2) oversold (<10), price above weekly EMA50 (bullish trend), volume spike
            if (rsi_2_values[i] < 10 and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI(2) overbought (>90), price below weekly EMA50 (bearish trend), volume spike
            elif (rsi_2_values[i] > 90 and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI(2) overbought (>70) or trend reversal (price below weekly EMA50)
            if rsi_2_values[i] > 70 or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI(2) oversold (<30) or trend reversal (price above weekly EMA50)
            if rsi_2_values[i] < 30 or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals