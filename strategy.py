# 4h_LongOnly_RSI_Momentum_Trend
# Strategy: Long-only strategy for 4h timeframe using RSI momentum and trend filters.
# Rationale: In both bull and bear markets, strong momentum moves often occur when RSI is above 50 and rising.
# We use a 4h RSI(14) > 50 and rising, combined with price above 4h EMA50 for trend confirmation.
# Volume confirmation is added to ensure conviction. We exit when RSI falls below 50 or trend breaks.
# This approach aims to capture momentum moves while avoiding excessive trading.
# Parameters are tuned to keep trade frequency within reasonable limits (target: 20-50 trades/year).
# Only long positions are taken to simplify and reduce whipsaw in choppy markets.

name = "4h_LongOnly_RSI_Momentum_Trend"
timeframe = "4h"
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
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h trend: EMA50
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_4h = close > ema_50
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    for i in range(20, n):
        # Get values
        rsi_val = rsi[i]
        rsi_prev = rsi[i-1] if i > 0 else 50
        uptrend = uptrend_4h[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: RSI > 50 and rising, uptrend, volume confirmation
            if rsi_val > 50 and rsi_val > rsi_prev and uptrend and vol_conf:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI < 50 or trend breaks down
            if rsi_val < 50 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals