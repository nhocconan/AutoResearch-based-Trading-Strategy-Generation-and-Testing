# 1d_KAMA_Trend_RSI_Pullback
# 1d strategy using KAMA trend filter, RSI pullback entries, and volume confirmation
# Long: KAMA trending up + RSI < 35 + volume > 1.5x 20-day avg
# Short: KAMA trending down + RSI > 65 + volume > 1.5x 20-day avg
# Exit: Opposite signal or trend reversal
# Designed for ~10-20 trades/year per symbol (40-80 total over 4 years)
# Works in bull trends (buy pullbacks) and bear trends (sell rallies)

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data (same as primary for 1d timeframe)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # KAMA components
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.zeros_like(close_1d, dtype=float)
    er[1:] = change[1:] / np.where(volatility[1:] == 0, 1, volatility[1:])
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # KAMA trend (slope)
    kama_slope = np.diff(kama, prepend=0)
    
    # RSI
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss == 0, 0, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # need for RSI and KAMA stability
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_slope[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = kama_slope[i] > 0
        downtrend = kama_slope[i] < 0
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        # RSI pullback conditions
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        
        if position == 0:
            # Long: uptrend + RSI oversold + volume
            if uptrend and rsi_oversold and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + RSI overbought + volume
            elif downtrend and rsi_overbought and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change or RSI overbought
            if not uptrend or rsi[i] > 65:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change or RSI oversold
            if not downtrend or rsi[i] < 35:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_RSI_Pullback"
timeframe = "1d"
leverage = 1.0