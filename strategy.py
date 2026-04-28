# 1d_KAMA_Direction_RSI14_Pullback_Volume_Spike
# Hypothesis: KAMA directional filter on daily timeframe + RSI(14) pullback entry + volume spike
# captures trend continuations in both bull and bear markets. KAMA adapts to volatility,
# reducing whipsaw in choppy markets. Volume confirms institutional participation.
# Target: 15-25 trades/year per symbol to minimize fee drag.

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly KAMA for trend filter
    close_1w = df_1w['close'].values
    # Efficiency ratio
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility = np.sum(np.abs(np.diff(close_1w)), axis=0)
    # Calculate ER for each point
    er = np.zeros_like(close_1w)
    for i in range(len(close_1w)):
        if i == 0:
            er[i] = 1.0
        else:
            if volatility[i] > 0:
                er[i] = change[i] / volatility[i]
            else:
                er[i] = 1.0
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    kama_1w = kama
    
    # Align KAMA to daily timeframe
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Daily RSI(14) for pullback entry
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for all indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # KAMA trend direction
        uptrend = close[i] > kama_1w_aligned[i]
        downtrend = close[i] < kama_1w_aligned[i]
        
        # RSI pullback conditions
        rsi_pullback_long = rsi[i] < 40  # Oversold pullback in uptrend
        rsi_pullback_short = rsi[i] > 60  # Overbought pullback in downtrend
        
        # Entry conditions with volume confirmation
        long_entry = uptrend and rsi_pullback_long and volume_spike[i]
        short_entry = downtrend and rsi_pullback_short and volume_spike[i]
        
        # Exit on trend reversal
        long_exit = not uptrend  # Trend turned down
        short_exit = not downtrend  # Trend turned up
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_KAMA_Direction_RSI14_Pullback_Volume_Spike"
timeframe = "1d"
leverage = 1.0