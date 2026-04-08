# 12h_kama_rsi_1w_trend_volume_v1
# Hypothesis: KAMA trend direction from 1-week + RSI pullback on 12h with volume confirmation captures trend-following entries during pullbacks. Uses KAMA's adaptive smoothing to reduce whipsaw in choppy markets. Weekly trend filter ensures alignment with higher timeframe momentum. Designed for low trade frequency (<30/year) to minimize fee drag in ranging 2025 market.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_kama_rsi_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for KAMA trend (primary trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # KAMA calculation: ER = |close - close[10]| / sum(|close - close[-1]|, 10)
    # SC = [ER * (fastest - slowest) + slowest]^2
    # KAMA = prevKAMA + SC * (price - prevKAMA)
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility = pd.Series(change).rolling(window=10, min_periods=10).sum().values
    direction = np.abs(np.subtract(close_1w, np.roll(close_1w, 10)))
    direction[:10] = 0  # first 10 values undefined
    er = np.where(volatility != 0, direction / volatility, 0)
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # Align 1w KAMA to 12h
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # RSI(14) on 12h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: 12h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(rsi[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI > 70 (overbought) OR price < KAMA (trend change)
            if (rsi[i] > 70) or (close[i] < kama_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 30 (oversold) OR price > KAMA (trend change)
            if (rsi[i] < 30) or (close[i] > kama_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: RSI < 35 (pullback in uptrend) + volume + price > KAMA
            if (rsi[i] < 35) and volume_filter[i] and (close[i] > kama_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: RSI > 65 (pullback in downtrend) + volume + price < KAMA
            elif (rsi[i] > 65) and volume_filter[i] and (close[i] < kama_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals