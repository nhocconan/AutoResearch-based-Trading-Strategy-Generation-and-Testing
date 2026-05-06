#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d RSI for momentum and 1w ATR for volatility regime
# - Uses 1w ATR percentile to identify low volatility regimes (squeeze)
# - Uses 1d RSI > 55 for bullish momentum and < 45 for bearish momentum
# - Enters long when price breaks above 1d high with volume spike in low vol + bullish momentum
# - Enters short when price breaks below 1d low with volume spike in low vol + bearish momentum
# - Exits when price crosses back below/above 1d close or volatility expands (ATR > 80th percentile)
# - Designed to capture volatility breakouts after weekly consolidation with daily momentum confirmation
# - Target: 80-160 total trades over 4 years (20-40/year) with 0.25 position sizing

name = "4h_1wATRPercentile_1dRSI_Breakout"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for 1d high/low and RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Get 1w data for ATR calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1d high and low for breakout levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d RSI (14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    avg_gain = wilders_smoothing(gain, 14)
    avg_loss = wilders_smoothing(loss, 14)
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1w ATR (14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    atr = wilders_smoothing(tr, 14)
    
    # Calculate 1w ATR percentile rank (lookback 50 periods)
    atr_series = pd.Series(atr)
    atr_percentile = atr_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Align 1d indicators to 4h timeframe
    high_1d_4h = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_4h = align_htf_to_ltf(prices, df_1d, low_1d)
    close_1d_4h = align_htf_to_ltf(prices, df_1d, close_1d)
    rsi_4h = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Align 1w ATR percentile to 4h timeframe
    atr_percentile_4h = align_htf_to_ltf(prices, df_1w, atr_percentile)
    
    # Volume filters (4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(high_1d_4h[i]) or np.isnan(low_1d_4h[i]) or np.isnan(close_1d_4h[i]) or
            np.isnan(rsi_4h[i]) or np.isnan(atr_percentile_4h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for low volatility regime (ATR < 20th percentile) and bullish/bearish momentum
            low_vol_regime = atr_percentile_4h[i] < 20
            
            if low_vol_regime:
                # Long: price breaks above 1d high with volume spike and RSI > 55
                if close[i] > high_1d_4h[i] and volume_spike[i] and rsi_4h[i] > 55:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below 1d low with volume spike and RSI < 45
                elif close[i] < low_1d_4h[i] and volume_spike[i] and rsi_4h[i] < 45:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price crosses below 1d close OR volatility expands (ATR > 80th percentile)
            if close[i] < close_1d_4h[i] or atr_percentile_4h[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1d close OR volatility expands (ATR > 80th percentile)
            if close[i] > close_1d_4h[i] or atr_percentile_4h[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals