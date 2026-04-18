#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Linear Regression Channel Breakout with Volume Confirmation and 1w Trend Filter
# Linear regression channel identifies the statistical trend and volatility envelope.
# Breakout beyond 2 standard deviations of the regression line with volume confirmation
# indicates strong momentum. 1-week trend filter ensures alignment with higher timeframe trend.
# Works in bull markets (upside breaks above upper channel) and bear markets (downside breaks below lower channel).
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.
name = "6h_LinearRegressionChannel_Breakout_Volume_1wTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate linear regression channel on close prices (60 periods = ~15 days)
    lookback = 60
    slope = np.zeros(n)
    intercept = np.zeros(n)
    upper_channel = np.zeros(n)
    lower_channel = np.zeros(n)
    
    # Calculate linear regression for each point
    for i in range(lookback, n):
        y = close[i-lookback:i]
        x = np.arange(lookback)
        if len(y) == lookback and not np.any(np.isnan(y)):
            # Linear regression: y = mx + b
            A = np.vstack([x, np.ones(len(x))]).T
            m, b = np.linalg.lstsq(A, y, rcond=None)[0]
            slope[i] = m
            intercept[i] = b
            # Calculate standard error of estimate
            y_pred = m * x + b
            residuals = y - y_pred
            std_err = np.sqrt(np.sum(residuals**2) / (lookback - 2))
            # Channel lines: y = mx + b ± 2*std_err
            upper_channel[i] = m * (lookback-1) + b + 2 * std_err
            lower_channel[i] = m * (lookback-1) + b - 2 * std_err
    
    # Align 1w trend (using 1w close price EMA50)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volume spike: current volume > 1.5 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 50)  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper_val = upper_channel[i]
        lower_val = lower_channel[i]
        ema_1w_val = ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: Close above upper channel AND price above 1w EMA50 AND volume spike
            if close_val > upper_val and close_val > ema_1w_val and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below lower channel AND price below 1w EMA50 AND volume spike
            elif close_val < lower_val and close_val < ema_1w_val and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close below lower channel (mean reversion) or above upper channel (take profit)
            if close_val < lower_val or close_val > upper_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close above upper channel (mean reversion) or below lower channel (take profit)
            if close_val > upper_val or close_val < lower_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals