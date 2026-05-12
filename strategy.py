# 4h_KAMA_Trend_RSI_MeanReversion
# Hypothesis: On 4h timeframe, use KAMA to detect trend direction and RSI for mean-reversion entries.
# In trending markets (KAMA slope aligned with price), buy RSI dips in uptrend, sell RSI rallies in downtrend.
# In ranging markets (KAMA flat), fade RSI extremes. Volume filter ensures institutional participation.
# Designed for low turnover: only trade when trend/momentum/volume align.

name = "4h_KAMA_Trend_RSI_MeanReversion"
timeframe = "4h"
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

    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')

    # KAMA on daily close: trend detection
    close_1d = df_1d['close'].values
    # Calculate Efficiency Ratio (ER) and Smoothing Constants
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.zeros_like(close_1d)
    er[1:] = change[1:] / (volatility[1:] + 1e-10)  # Avoid division by zero
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2  # Fast=2, Slow=30
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d = kama
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)

    # KAMA slope (trend strength)
    kama_slope = np.diff(kama_1d, prepend=0)
    kama_slope_aligned = align_htf_to_ltf(prices, df_1d, kama_slope)

    # RSI on 4h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Volume spike: current > 1.5x average of last 6 bars (1 day on 4h)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after RSI warmup
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(kama_slope_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # Determine market regime
            if abs(kama_slope_aligned[i]) > 0.001:  # Trending market
                # LONG: uptrend (kama rising) + RSI oversold + volume spike
                if (kama_slope_aligned[i] > 0 and 
                    rsi[i] < 35 and 
                    volume_spike[i]):
                    signals[i] = 0.25
                    position = 1
                # SHORT: downtrend (kama falling) + RSI overbought + volume spike
                elif (kama_slope_aligned[i] < 0 and 
                      rsi[i] > 65 and 
                      volume_spike[i]):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:  # Ranging market
                # LONG: RSI deeply oversold + volume spike
                if (rsi[i] < 30 and volume_spike[i]):
                    signals[i] = 0.25
                    position = 1
                # SHORT: RSI deeply overbought + volume spike
                elif (rsi[i] > 70 and volume_spike[i]):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI overbought or trend breaks
            if (rsi[i] > 65 or 
                kama_slope_aligned[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI oversold or trend breaks
            if (rsi[i] < 35 or 
                kama_slope_aligned[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals