#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Bollinger Band squeeze with daily mean reversion and volume confirmation.
# In low volatility regimes (BB width < 20th percentile), price tends to mean revert to the 20-day SMA.
# Long when price < lower BB and RSI < 30 with volume confirmation.
# Short when price > upper BB and RSI > 70 with volume confirmation.
# Weekly trend filter (BB width trend) ensures we only trade squeezes that are contracting.
# Designed for low trade frequency (10-20/year) to minimize whipsaw in ranging markets.

name = "1d_BBSqueeze_MeanReversion_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Bollinger Band calculation and squeeze detection
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Bollinger Bands (20, 2)
    close_1w_series = pd.Series(close_1w)
    bb_middle = close_1w_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_1w_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Weekly BB width percentile (20-day lookback) to detect squeeze
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 20 else np.nan, raw=False
    ).values
    
    # Squeeze condition: BB width < 20th percentile (low volatility)
    squeeze_condition = bb_width_percentile < 0.20
    
    # Align weekly data to daily timeframe
    bb_middle_aligned = align_htf_to_ltf(prices, df_1w, bb_middle)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1w, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1w, bb_lower)
    squeeze_aligned = align_htf_to_ltf(prices, df_1w, squeeze_condition.astype(float))
    
    # Daily RSI(14) for mean reversion signals
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values  # Fill NaN with neutral 50
    
    # Volume confirmation: current volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bb_middle_aligned[i]) or np.isnan(bb_upper_aligned[i]) or
            np.isnan(bb_lower_aligned[i]) or np.isnan(squeeze_aligned[i]) or
            np.isnan(rsi_values[i]) or np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: price at/below lower BB, RSI < 30, in squeeze, with volume
            if (squeeze_aligned[i] > 0.5 and  # In weekly BB squeeze
                close[i] <= bb_lower_aligned[i] * 1.002 and  # At or below lower BB
                rsi_values[i] < 30 and                    # Oversold RSI
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short setup: price at/above upper BB, RSI > 70, in squeeze, with volume
            elif (squeeze_aligned[i] > 0.5 and    # In weekly BB squeeze
                  close[i] >= bb_upper_aligned[i] * 0.998 and  # At or above upper BB
                  rsi_values[i] > 70 and                   # Overbought RSI
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses above middle BB or RSI > 50
            if close[i] >= bb_middle_aligned[i] or rsi_values[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses below middle BB or RSI < 50
            if close[i] <= bb_middle_aligned[i] or rsi_values[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals