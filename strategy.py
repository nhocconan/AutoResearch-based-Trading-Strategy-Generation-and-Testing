#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly Bollinger Band squeeze breakout with volume confirmation.
# In low volatility (BBW < 20th percentile): breakout in direction of weekly trend (EMA50).
# Uses volume > 1.5x 20-day average for confirmation. Avoids false breakouts in high volatility.
# Target: 10-25 trades/year by requiring Bollinger squeeze + breakout + volume + weekly trend alignment.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) == 0:
        return np.zeros(n)
    
    # Weekly EMA50 for trend direction
    weekly_close = df_weekly['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema50)
    
    # Daily Bollinger Bands (20, 2)
    close_series = prices['close']
    bb_mid = close_series.rolling(window=20, min_periods=20).mean()
    bb_std = close_series.rolling(window=20, min_periods=20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid
    
    # Bollinger Band width percentile (252-day lookback for 1-year)
    bb_width_percentile = bb_width.rolling(window=252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50,
        raw=False
    ).values
    
    # Pre-compute volume moving average (20-day)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(bb_width_percentile[i]) or np.isnan(vol_ma[i]) or np.isnan(weekly_ema50_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Bollinger squeeze: width below 20th percentile (low volatility)
        is_squeeze = bb_width_percentile[i] < 20.0
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # Weekly trend direction
        weekly_trend_up = price > weekly_ema50_aligned[i]
        weekly_trend_down = price < weekly_ema50_aligned[i]
        
        if position == 0:
            if is_squeeze and volume_confirm:
                # Breakout in direction of weekly trend
                if price > bb_upper.iloc[i] and weekly_trend_up:
                    signals[i] = 0.25
                    position = 1
                elif price < bb_lower.iloc[i] and weekly_trend_down:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions: mean reversion to middle band or volatility expansion
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns to middle band or volatility expands (squeeze ends)
                if price <= bb_mid.iloc[i] or bb_width_percentile[i] >= 50.0:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns to middle band or volatility expands
                if price >= bb_mid.iloc[i] or bb_width_percentile[i] >= 50.0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_BB_Squeeze_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0