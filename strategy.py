#!/usr/bin/env python3
"""
Hypothesis: 1h RSI(14) mean reversion with 4h EMA50 trend filter and session filter (08-20 UTC).
- Uses 4h EMA50 for trend direction (long only above, short only below)
- Enters on 1h RSI(14) < 30 for long, > 70 for short with volume confirmation (>1.5x 20-period average)
- Exits on RSI crossing back to 50 (mean reversion completion) or opposite extreme
- Session filter reduces noise trades during low-activity hours (00-08 and 20-24 UTC)
- Position size: 0.20 discrete level to minimize fee churn
- Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)
- Works in both bull/bear via trend filter + mean reversion in ranging markets
"""

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
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 4h EMA(50)
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 14)  # Volume MA, EMA, RSI
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(rsi_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only
        hour = prices.index[i].hour
        in_session = 8 <= hour <= 20
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # RSI conditions
        rsi_oversold = rsi_values[i] < 30
        rsi_overbought = rsi_values[i] > 70
        rsi_exit_long = rsi_values[i] > 50
        rsi_exit_short = rsi_values[i] < 50
        
        if position == 0 and in_session:
            # Long: RSI oversold AND price above 4h EMA50 AND volume confirmation
            if rsi_oversold and close[i] > ema_50_4h_aligned[i] and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought AND price below 4h EMA50 AND volume confirmation
            elif rsi_overbought and close[i] < ema_50_4h_aligned[i] and volume_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: RSI crosses back to 50 OR session ends
            if rsi_exit_long or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI crosses back to 50 OR session ends
            if rsi_exit_short or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI14_MeanReversion_4hEMA50_Trend_Volume_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0