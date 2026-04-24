#!/usr/bin/env python3
"""
Hypothesis: 1h RSI(14) mean reversion with 4h trend filter and volume spike confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h EMA(50) for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- RSI(14): oversold <30 for long, overbought >70 for short on 1h.
- Volume confirmation: current volume > 2.0 * 20-period volume MA to avoid chop.
- Entry: Long when RSI<30 AND 4h trend bullish AND volume confirmed.
         Short when RSI>70 AND 4h trend bearish AND volume confirmed.
- Exit: RSI crosses back to 50 (mean reversion completion).
- Signal size: 0.20 discrete to minimize fee churn and control drawdown.
Designed to work in both bull and bear markets via 4h trend filter and volatility-adjusted entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF EMA50 to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50, 14, 20)  # Need enough bars for EMA50, RSI, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi_values[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (2.0x threshold)
            vol_confirmed = curr_volume > 2.0 * vol_ma[i]
            
            # Determine 4h trend: bullish if close > EMA50, bearish if close < EMA50
            htf_close_aligned = align_htf_to_ltf(prices, df_4h, close_4h)
            htf_close = htf_close_aligned[i]
            
            trend_bullish = htf_close > ema_50_4h_aligned[i]
            trend_bearish = htf_close < ema_50_4h_aligned[i]
            
            # Long: RSI<30 AND 4h trend bullish AND volume confirmed
            if rsi_values[i] < 30 and trend_bullish and vol_confirmed:
                signals[i] = 0.20
                position = 1
            # Short: RSI>70 AND 4h trend bearish AND volume confirmed
            elif rsi_values[i] > 70 and trend_bearish and vol_confirmed:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long when RSI crosses back to 50 (mean reversion)
            if rsi_values[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short when RSI crosses back to 50 (mean reversion)
            if rsi_values[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI14_4hEMA50_Trend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0