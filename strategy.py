#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with RSI(14) mean reversion entries and volume confirmation.
# Uses KAMA(ER=10) for adaptive trend filtering, RSI(14)<30 for long and >70 for short,
# with volume > 1.5x 20-period average for confirmation. Designed for low trade frequency
# (~50 total trades over 4 years) to avoid fee drag. Works in bull/bear via KAMA trend filter
# and RSI mean reversion for precise entries during pullbacks.

name = "1d_KAMA10_RSI14_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate KAMA(ER=10) for trend filter
    close_series = pd.Series(close)
    change = np.abs(close_series.diff(10).values)
    volatility = np.abs(close_series.diff(1).rolling(window=10).sum().values)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14) for mean reversion entries
    delta = pd.Series(close).diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(10, 14, 20) + 1  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(kama[i]) or
            np.isnan(rsi[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_kama = kama[i]
        curr_rsi = rsi[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price > KAMA (uptrend), RSI < 30 (oversold), volume confirmation
            if (curr_close > curr_kama and 
                curr_rsi < 30 and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price < KAMA (downtrend), RSI > 70 (overbought), volume confirmation
            elif (curr_close < curr_kama and 
                  curr_rsi > 70 and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit: price < KAMA (trend change) or RSI > 50 (mean reversion complete)
            if curr_close < curr_kama or curr_rsi > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price > KAMA (trend change) or RSI < 50 (mean reversion complete)
            if curr_close > curr_kama or curr_rsi < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals