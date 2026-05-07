# 1h_RSI_Divergence_Trend_Confirmation
# Hypothesis: 1-hour RSI with bullish/bearish divergence detection combined with 4h trend filter (EMA50) and volume confirmation. 
# Works in bull markets: RSI bullish divergence during pullbacks in uptrend signals long entries.
# Works in bear markets: RSI bearish divergence during rallies in downtrend signals short entries.
# Volume confirmation ensures institutional participation, reducing false signals.
# Target: 15-35 trades/year per symbol by requiring confluence of RSI divergence, trend alignment, and volume spike.

#!/usr/bin/env python3
name = "1h_RSI_Divergence_Trend_Confirmation"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate RSI(14) on 1h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate volume moving average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for RSI and EMA stabilization
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike condition (current volume > 1.5x 20-period average)
        vol_spike = volume[i] > vol_ma[i] * 1.5
        
        # Check for RSI divergence (look back 5 periods for swing points)
        if i >= 5:
            # Recent price swings
            recent_high = np.max(high[i-4:i+1])
            recent_low = np.min(low[i-4:i+1])
            prev_high = np.max(high[i-9:i-4]) if i >= 9 else recent_high
            prev_low = np.min(low[i-9:i-4]) if i >= 9 else recent_low
            
            # RSI at corresponding points
            rsi_recent = rsi[i-4:i+1]
            rsi_prev = rsi[i-9:i-4] if i >= 9 else rsi_recent
            
            # Bullish divergence: price makes lower low, RSI makes higher low
            bull_div = (low[i] < low[i-4]) and (rsi[i] > rsi[i-4]) and (recent_low < prev_low)
            # Bearish divergence: price makes higher high, RSI makes lower high
            bear_div = (high[i] > high[i-4]) and (rsi[i] < rsi[i-4]) and (recent_high > prev_high)
        else:
            bull_div = bear_div = False
        
        if position == 0:
            # Long: RSI bullish divergence, price above 4h EMA50 (uptrend), volume spike
            if bull_div and close[i] > ema_50_4h_aligned[i] and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short: RSI bearish divergence, price below 4h EMA50 (downtrend), volume spike
            elif bear_div and close[i] < ema_50_4h_aligned[i] and vol_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: RSI bearish divergence or price below 4h EMA50
            if bear_div or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: RSI bullish divergence or price above 4h EMA50
            if bull_div or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# RSI divergence with trend filter and volume confirmation strategy.
# Enters on RSI divergence (bullish for longs, bearish for shorts) only when aligned with 4h trend.
# Volume spike confirms institutional participation.
# Exits on opposite divergence or trend violation.
# Position size 0.20 limits risk while allowing meaningful returns.
# Designed for 15-35 trades/year to minimize fee drag while capturing meaningful moves.