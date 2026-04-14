#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy combining 1-day ATR-based volatility breakout with 1-week RSI mean reversion.
# Long when price breaks above ATR-based upper band AND weekly RSI < 30 (oversold).
# Short when price breaks below ATR-based lower band AND weekly RSI > 70 (overbought).
# Exit when price returns to ATR-based middle band or weekly RSI crosses 50.
# Uses 1-day ATR for volatility-based channels (adapts to volatility regimes),
# 1-week RSI for mean-reversion filter, aiming to catch reversals in both bull and bear markets.
# Target: 15-35 trades/year per symbol (60-140 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load daily data ONCE for ATR-based volatility channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR(14) on daily data
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR-based channels: middle = close, upper = close + 1.5*ATR, lower = close - 1.5*ATR
    atr_middle = close_1d
    atr_upper = close_1d + 1.5 * atr
    atr_lower = close_1d - 1.5 * atr
    
    # Load weekly data ONCE for RSI mean reversion filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly RSI(14)
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align indicators to lower timeframe
    atr_upper_aligned = align_htf_to_ltf(prices, df_1d, atr_upper)
    atr_lower_aligned = align_htf_to_ltf(prices, df_1d, atr_lower)
    atr_middle_aligned = align_htf_to_ltf(prices, df_1d, atr_middle)
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(14, 14)  # Need ATR and RSI
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_upper_aligned[i]) or 
            np.isnan(atr_lower_aligned[i]) or
            np.isnan(atr_middle_aligned[i]) or
            np.isnan(rsi_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for volatility breakouts with RSI mean reversion filter
            # Long: price breaks above ATR upper AND weekly RSI < 30 (oversold)
            if (close[i] > atr_upper_aligned[i] and 
                rsi_aligned[i] < 30):
                position = 1
                signals[i] = position_size
            # Short: price breaks below ATR lower AND weekly RSI > 70 (overbought)
            elif (close[i] < atr_lower_aligned[i] and 
                  rsi_aligned[i] > 70):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to ATR middle or weekly RSI crosses above 50
            if (close[i] <= atr_middle_aligned[i] or 
                rsi_aligned[i] > 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to ATR middle or weekly RSI crosses below 50
            if (close[i] >= atr_middle_aligned[i] or 
                rsi_aligned[i] < 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_ATRBreakout_WeeklyRSI_MeanRev_v1"
timeframe = "12h"
leverage = 1.0