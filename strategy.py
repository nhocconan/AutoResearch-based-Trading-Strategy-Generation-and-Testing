#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w RSI for trend strength and 1w Bollinger Band width for volatility regime
# - Uses 1w Bollinger Band width percentile to identify low volatility regimes (squeeze)
# - Uses 1w RSI > 50 for bullish bias and < 50 for bearish bias
# - Enters long when price breaks above 1d high with volume spike in low vol + bullish bias
# - Enters short when price breaks below 1d low with volume spike in low vol + bearish bias
# - Exits when price crosses back below/above 1d close or volatility expands (BB width > 80th percentile)
# - Designed to capture volatility breakouts after weekly consolidation with weekly momentum confirmation
# - Target: 30-100 total trades over 4 years (7-25/year) with 0.25 position sizing

name = "1d_1wBBWidth_1wRSI_Breakout"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for 1d high/low
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Get 1w data for RSI and Bollinger Band width calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1d high and low for breakout levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1w RSI (14)
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
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
    
    # Calculate 1w Bollinger Bands (20, 2)
    # Middle band (SMA20)
    sma_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    # Standard deviation
    std_dev = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    # Upper and lower bands
    upper_bb = sma_20 + (2 * std_dev)
    lower_bb = sma_20 - (2 * std_dev)
    # Bollinger Band width
    bb_width = (upper_bb - lower_bb) / sma_20
    
    # Calculate 1w BB width percentile rank (lookback 50 periods)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Align 1d indicators to 1d timeframe (identity for same timeframe)
    high_1d_1d = high_1d
    low_1d_1d = low_1d
    close_1d_1d = close_1d
    
    # Align 1w indicators to 1d timeframe
    rsi_1d = align_htf_to_ltf(prices, df_1w, rsi)
    bb_width_percentile_1d = align_htf_to_ltf(prices, df_1w, bb_width_percentile)
    
    # Volume filters (1d timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(high_1d_1d[i]) or np.isnan(low_1d_1d[i]) or np.isnan(close_1d_1d[i]) or
            np.isnan(rsi_1d[i]) or np.isnan(bb_width_percentile_1d[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for low volatility regime (BB width < 20th percentile)
            low_vol_regime = bb_width_percentile_1d[i] < 20
            
            if low_vol_regime:
                # Bullish bias: RSI > 50
                if rsi_1d[i] > 50:
                    # Long: price breaks above 1d high with volume spike
                    if close[i] > high_1d_1d[i] and volume_spike[i]:
                        signals[i] = 0.25
                        position = 1
                # Bearish bias: RSI < 50
                elif rsi_1d[i] < 50:
                    # Short: price breaks below 1d low with volume spike
                    if close[i] < low_1d_1d[i] and volume_spike[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Exit long: price crosses below 1d close OR volatility expands (BB width > 80th percentile)
            if close[i] < close_1d_1d[i] or bb_width_percentile_1d[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1d close OR volatility expands (BB width > 80th percentile)
            if close[i] > close_1d_1d[i] or bb_width_percentile_1d[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals