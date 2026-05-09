#!/usr/bin/env python3
# Hypothesis: 1d Bollinger Band width regime + RSI mean reversion
# Long when BB width < 20th percentile (low volatility) and RSI < 30 (oversold)
# Short when BB width < 20th percentile (low volatility) and RSI > 70 (overbought)
# Exit when RSI crosses back to neutral (40-60 range) or BB width expands above 80th percentile
# Position size: 0.25 (25% of capital) to balance return and drawdown
# Designed to work in ranging markets via mean reversion and avoid trending markets via volatility filter

name = "1d_BB_Width_RSI_MeanReversion"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + bb_std * std
    lower = sma - bb_std * std
    bb_width = (upper - lower) / sma  # normalized width
    
    # Percentile ranks for BB width (using 50-period lookback)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else np.nan, raw=False
    ).values
    
    # RSI (14)
    rsi_period = 14
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Get 1w data for trend filter (optional, but we'll use it as regime filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        # Fallback to price vs 50-week EMA if not enough weekly data
        ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
        trend_filter = np.ones(n)  # neutral if no weekly data
    else:
        weekly_close = df_1w['close'].values
        ema50_weekly = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
        ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_weekly)
        # Trend filter: 1 if above weekly EMA50, -1 if below, 0 if undefined
        trend_filter = np.where(close > ema50_aligned, 1, 
                               np.where(close < ema50_aligned, -1, 0))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(sma[i]) or np.isnan(std[i]) or np.isnan(bb_width[i]) or
            np.isnan(bb_width_percentile[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filters: low volatility (BB width < 20th percentile) AND not strongly trending
        low_vol = bb_width_percentile[i] < 20
        not_strong_trend = abs(trend_filter[i]) < 1  # neutral or weak trend
        
        if position == 0:
            # Enter long: low volatility + oversold RSI
            if low_vol and not_strong_trend and rsi[i] < 30:
                signals[i] = 0.25
                position = 1
            # Enter short: low volatility + overbought RSI
            elif low_vol and not_strong_trend and rsi[i] > 70:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral OR volatility expands
            if rsi[i] > 40 or bb_width_percentile[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral OR volatility expands
            if rsi[i] < 60 or bb_width_percentile[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals