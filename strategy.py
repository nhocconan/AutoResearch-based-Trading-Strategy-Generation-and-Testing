#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h session-filtered (08-20 UTC) mean reversion using Bollinger Bands(20,2) with volume confirmation.
# Long when price touches lower BB AND volume > 1.5x 20-bar average AND RSI(14) < 30 (oversold).
# Short when price touches upper BB AND volume > 1.5x 20-bar average AND RSI(14) > 70 (overbought).
# Exit when price crosses BB middle band OR RSI returns to neutral (40-60).
# Uses 1h timeframe with session filter to reduce noise, targeting 60-150 trades over 4 years.
# Bollinger Bands provide dynamic support/resistance, volume confirms conviction, RSI avoids extreme overextension.

name = "1h_BollingerMeanReversion_VolumeRSI_Session"
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
    
    # Pre-compute session filter (08-20 UTC) ONCE before loop
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate Bollinger Bands(20,2) on 1h close
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    
    # Calculate RSI(14)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral RSI when undefined
    
    # Volume confirmation: >1.5x 20-bar average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (1.5 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after BB/RSI warmup
        # Skip if any value is NaN or outside session
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_middle[i]) or 
            np.isnan(rsi[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price at/below lower BB AND volume spike AND RSI oversold (<30)
            if (close[i] <= bb_lower[i] and 
                volume_spike[i] and 
                rsi[i] < 30):
                signals[i] = 0.20
                position = 1
            # Short: price at/above upper BB AND volume spike AND RSI overbought (>70)
            elif (close[i] >= bb_upper[i] and 
                  volume_spike[i] and 
                  rsi[i] > 70):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses above middle BB OR RSI returns to neutral (>40)
            if close[i] > bb_middle[i] or rsi[i] > 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses below middle BB OR RSI returns to neutral (<60)
            if close[i] < bb_middle[i] or rsi[i] < 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals