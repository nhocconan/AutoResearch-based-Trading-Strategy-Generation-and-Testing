#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily KAMA trend with 1-week EMA34 filter and volume confirmation.
# Uses daily KAMA for trend direction and 1-week EMA34 for higher timeframe trend alignment.
# Volume spike confirms trend strength. Designed to capture multi-week trends with low turnover.
# Target: 10-25 trades/year to stay within optimal range for 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for KAMA
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Get weekly data for EMA34
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate daily KAMA (efficiency ratio period=10)
    change = np.abs(np.diff(close_1d, 10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close_1d, 1)), axis=0)  # 10-period sum of absolute changes
    # Handle first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # smoothing constant
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if np.isnan(kama[i-1]):
            kama[i] = close_1d[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate weekly EMA34
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily KAMA and weekly EMA34 to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    ema34_1d = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: current volume > 2.0 * 50-period average
    volume_ma50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need KAMA and EMA34 warmup periods
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(ema34_1d[i]) or 
            np.isnan(volume_ma50[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 2.0x average (strict to reduce trades)
        volume_filter = volume[i] > (2.0 * volume_ma50[i])
        
        # Trend filters
        price_above_kama = close[i] > kama_aligned[i]
        price_below_kama = close[i] < kama_aligned[i]
        price_above_weekly_ema = close[i] > ema34_1d[i]
        price_below_weekly_ema = close[i] < ema34_1d[i]
        
        if position == 0:
            # Long: Price above daily KAMA and weekly EMA34 with volume confirmation
            if (price_above_kama and price_above_weekly_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price below daily KAMA and weekly EMA34 with volume confirmation
            elif (price_below_kama and price_below_weekly_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below daily KAMA OR weekly EMA34
            if (close[i] < kama_aligned[i]) or (close[i] < ema34_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above daily KAMA OR weekly EMA34
            if (close[i] > kama_aligned[i]) or (close[i] > ema34_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_With_1w_EMA34_Volume"
timeframe = "1d"
leverage = 1.0