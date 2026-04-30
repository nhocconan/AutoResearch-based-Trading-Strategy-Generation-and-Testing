#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h trend filter and volume confirmation.
# Long when RSI < 30 AND price > 4h EMA50 (uptrend) AND volume > 1.5x 20-bar average.
# Short when RSI > 70 AND price < 4h EMA50 (downtrend) AND volume > 1.5x 20-bar average.
# Exit when RSI crosses 50 (mean reversion complete) or after 12 bars max hold.
# Uses discrete position sizing (0.20) to limit drawdown and reduce fee churn.
# 4h EMA50 ensures we only trade with the dominant trend to avoid counter-trend entries in bear markets.
# Volume confirmation filters for institutional participation.
# Session filter (08-20 UTC) reduces noise during low-liquidity hours.
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.

name = "1h_RSI14_MeanReversion_4hEMA50_Trend_Volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate RSI(14) on 1h close
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # track holding period for time-based exit
    
    start_idx = max(50, 20)  # warmup for EMA and RSI calculations
    
    for i in range(start_idx, n):
        # Skip if not in trading session or indicators not available
        if not in_session[i] or \
           np.isnan(ema_50_4h_aligned[i]) or \
           np.isnan(rsi_values[i]) or \
           np.isnan(volume_confirm[i]):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        curr_close = close[i]
        curr_rsi = rsi_values[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            bars_since_entry = 0
            # Long: RSI < 30 (oversold), uptrend (price > 4h EMA50), volume confirmation
            if (curr_rsi < 30 and 
                curr_close > ema_50_4h_aligned[i] and 
                curr_volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: RSI > 70 (overbought), downtrend (price < 4h EMA50), volume confirmation
            elif (curr_rsi > 70 and 
                  curr_close < ema_50_4h_aligned[i] and 
                  curr_volume_confirm):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:  # Long position
            bars_since_entry += 1
            # Exit conditions: RSI crosses 50 (mean reversion) OR max hold of 12 bars reached
            if curr_rsi > 50 or bars_since_entry >= 12:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            bars_since_entry += 1
            # Exit conditions: RSI crosses 50 (mean reversion) OR max hold of 12 bars reached
            if curr_rsi < 50 or bars_since_entry >= 12:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.20
    
    return signals