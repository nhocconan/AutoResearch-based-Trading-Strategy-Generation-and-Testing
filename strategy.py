# 2025-06-24
# Hypothesis: On 4h timeframe, price reacts to 12-hour Fibonacci retracement levels (38.2%, 61.8%) of the prior 12-hour range.
# Combine with 12-hour EMA34 trend filter and volume confirmation to capture reversals at key levels.
# Long when price bounces off 61.8% retracement of prior 12h range with volume > 1.5x average and price above 12h EMA34.
# Short when price rejects at 38.2% retracement of prior 12h range with volume > 1.5x average and price below 12h EMA34.
# Exit on opposite retracement touch or return to 50% level.
# Designed for 4h to work in trending and ranging markets with ~20-40 trades per year.

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
    
    # Get 12h data for prior period's range
    df_12h = get_htf_data(prices, '12h')
    
    # Prior 12h high, low, and close (use shift(1) to avoid look-ahead)
    phigh = df_12h['high'].shift(1).values
    plow = df_12h['low'].shift(1).values
    pclose = df_12h['close'].values
    
    # Calculate 12h range and Fibonacci levels
    prange = phigh - plow
    p382 = plow + 0.382 * prange  # 38.2% retracement
    p50 = plow + 0.5 * prange     # 50% retracement
    p618 = plow + 0.618 * prange  # 61.8% retracement
    
    # Calculate 12h EMA34 for trend filter
    ema_34 = pd.Series(pclose).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all 12h levels to 4h timeframe
    phigh_4h = align_htf_to_ltf(prices, df_12h, phigh)
    plow_4h = align_htf_to_ltf(prices, df_12h, plow)
    p382_4h = align_htf_to_ltf(prices, df_12h, p382)
    p50_4h = align_htf_to_ltf(prices, df_12h, p50)
    p618_4h = align_htf_to_ltf(prices, df_12h, p618)
    ema_34_4h = align_htf_to_ltf(prices, df_12h, ema_34)
    
    # Volume confirmation: 20-period volume MA on 4h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for EMA34 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(phigh_4h[i]) or np.isnan(plow_4h[i]) or np.isnan(p382_4h[i]) or
            np.isnan(p50_4h[i]) or np.isnan(p618_4h[i]) or np.isnan(ema_34_4h[i]) or
            np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        if position == 0:
            # Long: price bounces off 61.8% retracement with volume confirmation and above EMA34
            if abs(price - p618_4h[i]) < 0.001 * price and vol > 1.5 * vol_ma and price > ema_34_4h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price rejects at 38.2% retracement with volume confirmation and below EMA34
            elif abs(price - p382_4h[i]) < 0.001 * price and vol > 1.5 * vol_ma and price < ema_34_4h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches 38.2% retracement (failed bounce) or returns to 50% level
            if abs(price - p382_4h[i]) < 0.001 * price or abs(price - p50_4h[i]) < 0.001 * price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches 61.8% retracement (failed rejection) or returns to 50% level
            if abs(price - p618_4h[i]) < 0.001 * price or abs(price - p50_4h[i]) < 0.001 * price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_FibRetracement_12H_EMA34_Volume"
timeframe = "4h"
leverage = 1.0