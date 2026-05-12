# 6h_1w_Stochastic_Divergence_Momentum
# Hypothesis: Weekly Stochastic RSI divergence with price on 6h chart signals exhaustion in both bull and bear markets. 
# Weekly timeframe reduces noise; Stochastic RSI captures overbought/oversold conditions with momentum.
# Divergence (price makes new high/low but Stochastic does not) indicates weakening trend and potential reversal.
# Works in bull markets (bearish divergence at tops) and bear markets (bullish divergence at bottoms) by fading extremes.
# Volume confirmation ensures institutional participation. Target: 15-25 trades/year.

name = "6h_1w_Stochastic_Divergence_Momentum"
timeframe = "6h"
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
    volume = prices['volume'].values

    # Get weekly data for Stochastic RSI
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)

    # Calculate weekly Stochastic RSI (14, 14, 3, 3)
    rsi_period = 14
    stoch_period = 14
    k_period = 3
    d_period = 3
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # RSI calculation
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Stochastic of RSI
    rsi_min = pd.Series(rsi).rolling(window=stoch_period, min_periods=stoch_period).min().values
    rsi_max = pd.Series(rsi).rolling(window=stoch_period, min_periods=stoch_period).max().values
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100
    
    # %K and %D
    k = pd.Series(stoch_rsi).rolling(window=k_period, min_periods=k_period).mean().values
    d = pd.Series(k).rolling(window=d_period, min_periods=d_period).mean().values
    
    # Align weekly Stochastic RSI to 6h
    k_aligned = align_htf_to_ltf(prices, df_1w, k)
    d_aligned = align_htf_to_ltf(prices, df_1w, d)
    
    # Weekly trend filter: price vs 50 EMA
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume confirmation: current volume > 1.3x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Track weekly high/low for divergence detection
    week_high = np.full_like(close_1w, np.nan)
    week_low = np.full_like(close_1w, np.nan)
    
    for i in range(1, len(close_1w)):
        week_high[i] = max(week_high[i-1], high_1w[i])
        week_low[i] = min(week_low[i-1], low_1w[i])
    week_high[0] = high_1w[0]
    week_low[0] = low_1w[0]
    
    week_high_aligned = align_htf_to_ltf(prices, df_1w, week_high)
    week_low_aligned = align_htf_to_ltf(prices, df_1w, week_low)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(k_aligned[i]) or np.isnan(d_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(volume_ok[i]) or np.isnan(week_high_aligned[i]) or np.isnan(week_low_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Bullish divergence: price makes new low but Stochastic does not
        bullish_div = (close[i] <= week_low_aligned[i] and 
                      k[i] > np.nanmin(k[max(0, i-10):i+1]) if not np.isnan(k[i]) else False)
        
        # Bearish divergence: price makes new high but Stochastic does not
        bearish_div = (close[i] >= week_high_aligned[i] and 
                      k[i] < np.nanmax(k[max(0, i-10):i+1]) if not np.isnan(k[i]) else False)
        
        if position == 0:
            # LONG: Bullish divergence with price below weekly EMA50 (bearish context) and volume
            if bullish_div and close[i] < ema_50_aligned[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish divergence with price above weekly EMA50 (bullish context) and volume
            elif bearish_div and close[i] > ema_50_aligned[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish divergence or price crosses above EMA50
            if bearish_div or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish divergence or price crosses below EMA50
            if bullish_div or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals