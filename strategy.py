#!/usr/bin/env python3
"""
Hypothesis: 1h RSI(14) mean reversion with 4h trend filter and volume confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h for EMA200 trend filter and 1d for ATR-based volume spike confirmation.
- Entry: Long when RSI(14) < 30 AND price > 4h EMA200 AND volume spike (current ATR/20-period ATR > 1.5).
         Short when RSI(14) > 70 AND price < 4h EMA200 AND volume spike.
- Exit: RSI crosses back to neutral (40-60 range) or opposite RSI extreme.
- Signal size: 0.20 discrete to minimize fee drag while maintaining profit potential.
- Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
- Session filter: 08-20 UTC to avoid low-volume Asian session noise.
- Estimated trades: ~100 total over 4 years (~25/year) based on RSI extreme frequency with strict filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def rsi(close, period=14):
    """Calculate Relative Strength Index."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def atr(high, low, close, period):
    """Calculate Average True Range."""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]
    return pd.Series(true_range).ewm(span=period, adjust=False, min_periods=period).mean().values

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h trend filter: EMA200
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 100:
        return np.zeros(n)
    
    ema200_4h = ema(df_4h['close'].values, 200)
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # Calculate 1d ATR for volume spike confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 1h RSI
    rsi_values = rsi(close, 14)
    
    # Session filter: 08-20 UTC (avoid low-volume Asian session)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 200  # Need sufficient data for 4h EMA200
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if data not ready
        if (np.isnan(ema200_4h_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(rsi_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_rsi = rsi_values[i]
        
        # Exit conditions: RSI returns to neutral range (40-60) or opposite extreme
        if position != 0:
            # Exit long: RSI >= 40 or RSI >= 70 (overbought)
            if position == 1:
                if curr_rsi >= 40:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: RSI <= 60 or RSI <= 30 (oversold)
            elif position == -1:
                if curr_rsi <= 60:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: RSI extreme with trend filter and volume confirmation
        if position == 0:
            # Long: RSI < 30 (oversold) AND bullish 4h trend AND volume spike
            if curr_rsi < 30 and curr_close > ema200_4h_aligned[i] and atr_ratio_aligned[i] > 1.5:
                signals[i] = 0.20
                position = 1
            # Short: RSI > 70 (overbought) AND bearish 4h trend AND volume spike
            elif curr_rsi > 70 and curr_close < ema200_4h_aligned[i] and atr_ratio_aligned[i] > 1.5:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.20
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.20
    
    return signals

name = "1h_RSIMeanReversion_4hEMA200_TrendFilter_1dATR_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0