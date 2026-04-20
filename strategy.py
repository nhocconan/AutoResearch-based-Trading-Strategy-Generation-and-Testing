# ANALYSIS: This strategy combines 12h trend following with 4h momentum entries and volume confirmation.
# The 12h EMA50 provides the primary trend direction, reducing false signals during counter-trend moves.
# The 4h RSI provides momentum confirmation, and volume spikes confirm institutional participation.
# This approach should work in both bull and bear markets by following the higher timeframe trend.
# Target: 20-40 trades per year to minimize fee drag while capturing significant moves.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_EMA50_RSI_Volume_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop for EMA50 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # === 12h EMA50 for trend direction ===
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # === 4h RSI for momentum ===
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    loss_ma = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = gain_ma / np.where(loss_ma > 0, loss_ma, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    # === 4h Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        ema_val = ema_50_aligned[i]
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_val) or np.isnan(rsi_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above EMA50 (uptrend) + RSI > 55 (bullish momentum) + volume spike
            if close_val > ema_val and rsi_val > 55 and vol_ratio_val > 1.8:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: price below EMA50 (downtrend) + RSI < 45 (bearish momentum) + volume spike
            elif close_val < ema_val and rsi_val < 45 and vol_ratio_val > 1.8:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        
        elif position == 1:
            # Long exit: trend reversal or momentum fade
            if close_val <= ema_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal or momentum fade
            if close_val >= ema_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals