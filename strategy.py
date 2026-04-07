#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily KAMA trend with weekly RSI filter and volume confirmation
# Uses KAMA (Kaufman Adaptive Moving Average) to capture trend direction with lower lag in volatile markets
# Weekly RSI filter avoids counter-trend trades in strong weekly trends
# Volume confirmation ensures institutional participation
# Designed for low frequency (target: 10-25 trades/year) to minimize fee impact
# Works in bull markets via trend following, in bear markets via avoiding false signals during downtrends

name = "daily_kama_weekly_rsi_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly trend filter (RSI)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate weekly RSI(14)
    delta_1w = np.diff(close_1w, prepend=close_1w[0])
    gain_1w = np.where(delta_1w > 0, delta_1w, 0)
    loss_1w = np.where(delta_1w < 0, -delta_1w, 0)
    avg_gain_1w = pd.Series(gain_1w).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss_1w = pd.Series(loss_1w).ewm(alpha=1/14, adjust=False).mean().values
    rs_1w = avg_gain_1w / (avg_loss_1w + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs_1w))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Daily KAMA trend
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10, prepend=close[:10]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)
    # Fix dimensions: volatility should be same length as change
    volatility = pd.Series(volume).rolling(window=10, min_periods=1).sum().values  # proxy for volatility
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    # Alternative vectorized approach for stability
    kama = pd.Series(close).ewm(alpha=sc, adjust=False).mean().values
    
    # Volume confirmation (20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(rsi_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Trend direction from KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # Weekly trend filter: avoid extremes
        rsi_overbought = rsi_1w_aligned[i] > 70
        rsi_oversold = rsi_1w_aligned[i] < 30
        rsi_neutral = (rsi_1w_aligned[i] >= 30) & (rsi_1w_aligned[i] <= 70)
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit on trend reversal or weekly overbought
            if (close[i] < kama[i]) or rsi_overbought:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit on trend reversal or weekly oversold
            if (close[i] > kama[i]) or rsi_oversold:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long in uptrend with weekly not overbought
            if price_above_kama and (not rsi_overbought) and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short in downtrend with weekly not oversold
            elif price_below_kama and (not rsi_oversold) and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals