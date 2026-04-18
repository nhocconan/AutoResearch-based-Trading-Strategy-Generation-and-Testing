# NEW STRATEGY: 4h_KAMA_Trend_Filter_V1
# Hypothesis: KAMA adapts to market noise - in trending markets it follows price closely,
# in ranging markets it stays flat. Combined with volume confirmation and RSI filter,
# this should capture trends while avoiding whipsaws in ranging markets.
# Timeframe: 4h (optimal balance of signal quality and trade frequency)
# Works in bull markets: KAMA follows uptrend, enters on pullbacks with volume
# Works in bear markets: KAMA follows downtrend, enters on bounces with volume
# Expected trades: 20-40/year (80-160 total over 4 years) - avoids fee drag

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
    
    # KAMA calculation (adaptive moving average)
    # Efficiency Ratio = |Price Change| / Sum of Absolute Daily Changes
    change = np.abs(np.diff(close, prepend=close[0]))
    direction = np.abs(np.subtract(close, np.roll(close, 1)))
    direction[0] = np.abs(close[0] - close[0])  # first element
    
    # Avoid division by zero
    er = np.where(change != 0, direction / change, 0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI for overbought/oversold conditions
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)  # when no losses, RSI=100
    rsi = np.where(avg_gain == 0, 0, rsi)    # when no gains, RSI=0
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # KAMA needs warmup, RSI needs 14, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.2 * vol_ma[i]
        
        if position == 0:
            # Long: price above KAMA (uptrend) + RSI not overbought + volume confirmation
            if close[i] > kama[i] and rsi[i] < 70 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend) + RSI not oversold + volume confirmation
            elif close[i] < kama[i] and rsi[i] > 30 and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below KAMA (trend change) or RSI overbought
            if close[i] < kama[i] or rsi[i] > 75:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above KAMA (trend change) or RSI oversold
            if close[i] > kama[i] or rsi[i] < 25:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Trend_Filter_V1"
timeframe = "4h"
leverage = 1.0