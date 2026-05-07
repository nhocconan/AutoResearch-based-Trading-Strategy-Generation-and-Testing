#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly RSI Divergence with 1w EMA50 Trend Filter and Volume Confirmation
# Long when: Weekly RSI bullish divergence (price makes lower low, RSI makes higher low) AND price > weekly EMA50 AND volume > 1.5x 20-day average
# Short when: Weekly RSI bearish divergence (price makes higher high, RSI makes lower high) AND price < weekly EMA50 AND volume > 1.5x 20-day average
# Exit when weekly RSI returns to neutral zone (40-60) or volume drops below average
# Designed for 1d timeframe with low trade frequency (target: 10-25/year) to minimize fee drag
# Uses weekly timeframe for divergence detection to capture major reversals
# Volume filter ensures institutional participation
# RSI divergence works in both bull and bear markets by identifying exhaustion

name = "1d_WeeklyRSI_Divergence_EMA50_Volume"
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
    
    # Weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Weekly RSI for divergence detection
    rsi_period = 14
    delta = pd.Series(close_1w).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_values)
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for weekly indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Need at least 2 weekly points for divergence check
            if i >= 50 + 16:  # Need 2 weekly bars (approximately 16 daily bars per week)
                # Get current and previous weekly aligned values
                rsi_now = rsi_1w_aligned[i]
                rsi_prev = rsi_1w_aligned[i-16] if i-16 >= 0 else rsi_1w_aligned[0]
                price_now = close[i]
                price_prev = close[i-16] if i-16 >= 0 else close[0]
                
                # Bullish divergence: price makes lower low, RSI makes higher low
                bull_div = (price_now < price_prev) and (rsi_now > rsi_prev)
                # Bearish divergence: price makes higher high, RSI makes lower high
                bear_div = (price_now > price_prev) and (rsi_now < rsi_prev)
                
                if bull_div and (price_now > ema50_1w_aligned[i]) and volume_filter[i]:
                    signals[i] = 0.25
                    position = 1
                elif bear_div and (price_now < ema50_1w_aligned[i]) and volume_filter[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: RSI returns to neutral (>50) or volume filter fails
            if rsi_1w_aligned[i] > 50 or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI returns to neutral (<50) or volume filter fails
            if rsi_1w_aligned[i] < 50 or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals