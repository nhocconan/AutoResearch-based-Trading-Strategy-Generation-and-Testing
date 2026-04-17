#!/usr/bin/env python3
"""
4h_1d_RSI_Reversal_With_Filter
Strategy: Mean reversion on RSI extremes with trend and volatility filters.
Long: RSI(14) < 30 + price above 4h EMA20 + ATR(14) < 1.5x ATR(50) (low vol)
Short: RSI(14) > 70 + price below 4h EMA20 + ATR(14) < 1.5x ATR(50)
Exit: RSI crosses back to 50 (mean reversion complete)
Position size: 0.25
Designed to capture reversals in overbought/oversold conditions during low volatility periods,
which occur in both bull and bear markets. Filters prevent whipsaws in strong trends.
Timeframe: 4h
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to alpha=1/14)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, 0.0), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate EMA20 on 4h
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate ATR(14) and ATR(50) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # first bar
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volatility filter: current ATR < 1.5 * long-term ATR (avoid high volatility whipsaws)
    vol_filter = atr14 < (1.5 * atr50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough data for ATR50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(rsi[i]) or np.isnan(ema20[i]) or np.isnan(vol_filter[i]):
            signals[i] = 0.0
            continue
        
        # RSI conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_neutral = (rsi[i] >= 40) & (rsi[i] <= 60)  # exit zone
        
        # Trend filter: price relative to EMA20
        price_above_ema = close[i] > ema20[i]
        price_below_ema = close[i] < ema20[i]
        
        if position == 0:
            # Long: RSI oversold + price above EMA + low volatility
            if rsi_oversold and price_above_ema and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought + price below EMA + low volatility
            elif rsi_overbought and price_below_ema and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral zone (mean reversion)
            if rsi_neutral:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral zone
            if rsi_neutral:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_RSI_Reversal_With_Filter"
timeframe = "4h"
leverage = 1.0