#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion within 4h/1d trend using RSI extremes + volume spike
# Uses 4h EMA50 and 1d EMA200 for trend filter (long only above both EMA, short only below)
# Entry: RSI(14) < 30 for long, > 70 for short with volume > 1.5x 20-period average
# Exit: RSI returns to neutral zone (40-60) or trend filter fails
# Session filter: 08-20 UTC to avoid low-liquidity hours
# Position size: 0.20 discrete to minimize fee churn
# Target: 60-120 trades over 4 years (15-30/year) for 1h timeframe
# Works in bull/bear by only trading with higher timeframe trend

name = "1h_RSI_MeanReversion_4h1dTrend_Filter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d EMA200
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate RSI(14) on 1h close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(rsi_values[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: RSI < 30 (oversold) + volume spike + price > both EMAs
            if (rsi_values[i] < 30 and volume_spike[i] and 
                close[i] > ema50_4h_aligned[i] and close[i] > ema200_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI > 70 (overbought) + volume spike + price < both EMAs
            elif (rsi_values[i] > 70 and volume_spike[i] and 
                  close[i] < ema50_4h_aligned[i] and close[i] < ema200_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: RSI returns to neutral (40-60) or trend filter fails
            if (rsi_values[i] >= 40 and rsi_values[i] <= 60) or \
               close[i] <= ema50_4h_aligned[i] or close[i] <= ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral (40-60) or trend filter fails
            if (rsi_values[i] >= 40 and rsi_values[i] <= 60) or \
               close[i] >= ema50_4h_aligned[i] or close[i] >= ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals