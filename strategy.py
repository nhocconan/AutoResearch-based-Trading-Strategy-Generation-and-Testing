#!/usr/bin/env python3
# 1d_1w_RSI_Bollinger_Squeeze_Reversal
# Hypothesis: On daily timeframe, combine RSI(2) extremes with Bollinger Bands squeeze
# and weekly trend filter to capture mean-reversion bounces in both bull and bear markets.
# Weekly EMA50 filter ensures trades align with higher timeframe trend.
# Bollinger squeeze (bandwidth < 20th percentile) identifies low volatility periods
# preceding explosive moves, increasing reversal probability.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_RSI_Bollinger_Squeeze_Reversal"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for RSI and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === Daily RSI(2) ===
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def rsi_wilder(series, period):
        rsi = np.zeros_like(series)
        avg_gain = np.zeros_like(series)
        avg_loss = np.zeros_like(series)
        
        # Initial values
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period+1, len(series)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
            rs = avg_gain[i] / np.where(avg_loss[i] != 0, avg_loss[i], 1e-10)
            rsi[i] = 100 - (100 / (1 + rs))
        
        return rsi
    
    rsi_2 = rsi_wilder(close_1d, 2)
    
    # === Daily Bollinger Bands (20, 2) ===
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    bb_width = bb_upper - bb_lower
    
    # Bollinger squeeze: bandwidth below 20th percentile
    bb_width_series = pd.Series(bb_width)
    bb_width_pct = bb_width_series.rolling(window=50, min_periods=20).quantile(0.2).values
    squeeze = bb_width < bb_width_pct
    
    # === Weekly EMA50 for trend filter ===
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all daily and weekly indicators to daily timeframe
    rsi_2_aligned = align_htf_to_ltf(prices, df_1d, rsi_2)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze.astype(float))
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Get values
        close_val = prices['close'].iloc[i]
        rsi_val = rsi_2_aligned[i]
        bb_upper_val = bb_upper_aligned[i]
        bb_lower_val = bb_lower_aligned[i]
        squeeze_val = squeeze_aligned[i]
        ema50_val = ema50_1w_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi_val) or np.isnan(bb_upper_val) or np.isnan(bb_lower_val) or 
            np.isnan(squeeze_val) or np.isnan(ema50_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI(2) < 10 (extreme oversold) + Bollinger squeeze + price near lower BB + weekly uptrend
            if (rsi_val < 10 and 
                squeeze_val > 0.5 and  # Squeeze condition active
                close_val <= bb_lower_val * 1.02 and  # Near or below lower BB
                close_val > ema50_val):  # Only long in weekly uptrend
                signals[i] = 0.25
                position = 1
            # Short: RSI(2) > 90 (extreme overbought) + Bollinger squeeze + price near upper BB + weekly downtrend
            elif (rsi_val > 90 and 
                  squeeze_val > 0.5 and  # Squeeze condition active
                  close_val >= bb_upper_val * 0.98 and  # Near or above upper BB
                  close_val < ema50_val):  # Only short in weekly downtrend
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI(2) > 50 (mean reversion complete) or price reaches middle BB
            sma_20_val = sma_20[i]  # Already aligned via close price alignment
            if (rsi_val > 50 or 
                close_val >= sma_20_val * 0.995):  # Near middle BB
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI(2) < 50 (mean reversion complete) or price reaches middle BB
            sma_20_val = sma_20[i]  # Already aligned via close price alignment
            if (rsi_val < 50 or 
                close_val <= sma_20_val * 1.005):  # Near middle BB
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals