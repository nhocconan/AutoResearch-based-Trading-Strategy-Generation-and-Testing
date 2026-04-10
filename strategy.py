#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend direction + RSI(2) mean reversion + 1w volume confirmation
# - KAMA(10,2,30) determines primary trend direction on daily timeframe
# - RSI(2) < 10 for long entry, RSI(2) > 90 for short entry (extreme mean reversion)
# - 1w volume > 1.5x 20-period average confirms institutional participation
# - Only trade in direction of KAMA trend to avoid fighting the trend
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - Works in bull markets via trend continuation, in bear markets via mean reversion within trend

name = "1d_1w_kama_rsi2_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Primary data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d KAMA (Trend Direction) ===
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # 10-period volatility
    # Handle first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    
    # KAMA calculation
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # Start after first 10 periods
    for i in range(10, n):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA trend: 1 = above KAMA (uptrend), -1 = below KAMA (downtrend)
    kama_trend = np.where(close > kama, 1, -1)
    
    # === 1d RSI(2) (Mean Reversion) ===
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1w Volume Confirmation ===
    vol_ma = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_spike = df_1w['volume'].values > (1.5 * vol_ma)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1w, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(kama_trend[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_spike_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: RSI(2) < 10 (oversold) AND KAMA uptrend AND 1w volume spike
            if (rsi[i] < 10 and 
                kama_trend[i] == 1 and 
                vol_spike_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: RSI(2) > 90 (overbought) AND KAMA downtrend AND 1w volume spike
            elif (rsi[i] > 90 and 
                  kama_trend[i] == -1 and 
                  vol_spike_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when RSI returns to neutral (50) or opposite extreme
            exit_long = (position == 1 and 
                        (rsi[i] >= 50 or rsi[i] > 80))
            exit_short = (position == -1 and 
                         (rsi[i] <= 50 or rsi[i] < 20))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals