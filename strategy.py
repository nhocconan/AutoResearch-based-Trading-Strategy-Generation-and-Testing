#!/usr/bin/env python3
# 1h_4h1d_Momentum_Divergence_Filter
# Hypothesis: 1h momentum divergence with 4h trend filter and volume confirmation.
# Uses 4h RSI divergence (bullish/bearish) to identify reversals, confirmed by 1h price action and volume spike.
# 4h trend (close > EMA50) filters counter-trend signals. Designed for 15-30 trades/year per symbol.
# Works in bull/bear by requiring alignment with higher timeframe trend and momentum exhaustion signals.

name = "1h_4h1d_Momentum_Divergence_Filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def rsi(close, period=14):
    """Calculate RSI with proper handling of edge cases."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data for trend and RSI divergence
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 1h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h RSI for divergence detection
    rsi_4h = rsi(df_4h['close'].values, period=14)
    
    # Find 4h swing highs and lows for divergence
    # Swing high: high > previous 2 and next 2 highs
    # Swing low: low < previous 2 and next 2 lows
    swing_high = np.zeros(len(df_4h), dtype=bool)
    swing_low = np.zeros(len(df_4h), dtype=bool)
    
    for i in range(2, len(df_4h) - 2):
        if (df_4h['high'].values[i] > df_4h['high'].values[i-1] and 
            df_4h['high'].values[i] > df_4h['high'].values[i-2] and
            df_4h['high'].values[i] > df_4h['high'].values[i+1] and
            df_4h['high'].values[i] > df_4h['high'].values[i+2]):
            swing_high[i] = True
            
        if (df_4h['low'].values[i] < df_4h['low'].values[i-1] and 
            df_4h['low'].values[i] < df_4h['low'].values[i-2] and
            df_4h['low'].values[i] < df_4h['low'].values[i+1] and
            df_4h['low'].values[i] < df_4h['low'].values[i+2]):
            swing_low[i] = True
    
    # Detect bullish and bearish RSI divergence
    bullish_div = np.zeros(len(df_4h), dtype=bool)
    bearish_div = np.zeros(len(df_4h), dtype=bool)
    
    # Bullish divergence: price makes lower low, RSI makes higher low
    for i in range(4, len(df_4h)):
        if swing_low[i]:
            # Look back for previous swing low
            for j in range(i-1, max(0, i-20), -1):
                if swing_low[j]:
                    if (df_4h['low'].values[i] < df_4h['low'].values[j] and  # lower low
                        rsi_4h[i] > rsi_4h[j]):  # higher low on RSI
                        bullish_div[i] = True
                    break
    
    # Bearish divergence: price makes higher high, RSI makes lower high
    for i in range(4, len(df_4h)):
        if swing_high[i]:
            # Look back for previous swing high
            for j in range(i-1, max(0, i-20), -1):
                if swing_high[j]:
                    if (df_4h['high'].values[i] > df_4h['high'].values[j] and  # higher high
                        rsi_4h[i] < rsi_4h[j]):  # lower high on RSI
                        bearish_div[i] = True
                    break
    
    # Align 4h indicators to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    bullish_div_aligned = align_htf_to_ltf(prices, df_4h, bullish_div.astype(float))
    bearish_div_aligned = align_htf_to_ltf(prices, df_4h, bearish_div.astype(float))
    close_4h_aligned = align_htf_to_ltf(prices, df_4h, df_4h['close'].values)
    
    # Volume confirmation (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough history for 4h indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(bullish_div_aligned[i]) or
            np.isnan(bearish_div_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 4h trend: close > EMA50
        uptrend = close_4h_aligned[i] > ema_50_4h_aligned[i]
        downtrend = close_4h_aligned[i] < ema_50_4h_aligned[i]
        
        # Volume confirmation (2x average for significance)
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Bullish RSI divergence in uptrend with volume spike
            if bullish_div_aligned[i] > 0.5 and uptrend and volume_surge:
                signals[i] = 0.20
                position = 1
            # Short: Bearish RSI divergence in downtrend with volume spike
            elif bearish_div_aligned[i] > 0.5 and downtrend and volume_surge:
                signals[i] = -0.20
                position = -1
        else:
            if position == 1:
                # Long exit: bearish divergence appears or trend fails
                if bearish_div_aligned[i] > 0.5 or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Short exit: bullish divergence appears or trend fails
                if bullish_div_aligned[i] > 0.5 or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals