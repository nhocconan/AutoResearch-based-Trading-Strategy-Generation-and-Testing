# 1d_KAMA_Direction_RSI_Filter
# Hypothesis: Daily KAMA adapts to market efficiency, providing smooth trend direction.
# Combined with RSI for momentum confirmation and volatility filter to avoid chop.
# Works in bull/bear by requiring KAMA trend alignment with RSI momentum.
# Target: 15-25 trades/year per symbol with clean entries.

name = "1d_KAMA_Direction_RSI_Filter"
timeframe = "1d"
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
    
    # Weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily KAMA (adaptive moving average)
    # Efficiency Ratio = |Price Change| / Sum|Price Changes|
    change = np.abs(np.diff(close, prepend=close[0]))
    direction = np.abs(np.subtract(close, np.roll(close, 1)))
    er = np.divide(direction, change, out=np.zeros_like(direction), where=change!=0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Daily RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volatility filter: ATR(14) < 50th percentile of ATR(50) to avoid high chop
    tr1 = np.subtract(high, low)
    tr2 = np.subtract(np.abs(high), np.abs(np.roll(close, 1)))
    tr3 = np.subtract(np.abs(low), np.abs(np.roll(close, 1)))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).median().values
    low_vol = atr < atr_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA, RSI > 50, weekly uptrend, low volatility
            if (close[i] > kama[i] and 
                rsi[i] > 50 and 
                close[i] > ema_50_1w_aligned[i] and 
                low_vol[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI < 50, weekly downtrend, low volatility
            elif (close[i] < kama[i] and 
                  rsi[i] < 50 and 
                  close[i] < ema_50_1w_aligned[i] and 
                  low_vol[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below KAMA or RSI < 40
            if (close[i] < kama[i]) or (rsi[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above KAMA or RSI > 60
            if (close[i] > kama[i]) or (rsi[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals