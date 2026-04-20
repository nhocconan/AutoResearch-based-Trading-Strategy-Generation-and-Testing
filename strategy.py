# 12h_1w_Momentum_Reversal_v1
# Momentum reversal using 1w RSI extremes and 12h price action
# Works in bull/bear: buys oversold dips in uptrend, sells overbought rallies in downtrend
# 1w RSI defines overbought/oversold extremes, 12h price action confirms reversal
# Volume filter ensures conviction
# Target: 25-35 trades/year per symbol

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_Momentum_Reversal_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # === 1w: RSI(14) for overbought/oversold ===
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss > 0, avg_loss, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_1w = np.where(np.isnan(rs), 50, rsi)  # neutral when undefined
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # === 12h: Price and volume ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Get values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        rsi_val = rsi_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if np.isnan(rsi_val) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Price action signals
        prev_close = close[i-1]
        
        if position == 0:
            # Long: RSI oversold (<30) AND price closes above prior close (bullish reversal)
            if (rsi_val < 30 and close_val > prev_close and vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) AND price closes below prior close (bearish reversal)
            elif (rsi_val > 70 and close_val < prev_close and vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral (>50) or momentum fails
            if (rsi_val > 50 or close_val < prev_close):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral (<50) or momentum fails
            if (rsi_val < 50 or close_val > prev_close):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals