#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d RSI(14) mean reversion with 1w EMA50 trend filter and volume confirmation.
# Long when RSI crosses above 30 from below (oversold bounce) AND 1d volume > 1.5x 20-period average AND price > 1w EMA50.
# Short when RSI crosses below 70 from above (overbought rejection) AND 1d volume > 1.5x 20-period average AND price < 1w EMA50.
# Exit when RSI crosses back to 50 (neutral) to capture mean reversion in ranging markets.
# Uses 1w EMA50 for trend filter to avoid counter-trend trades.
# Target: 20-60 total trades over 4 years (5-15/year) for low fee drift.

name = "1d_RSI14_1wEMA50_Volume"
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
    
    # 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate RSI(14) on 1d data
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # 1d volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for RSI and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: RSI crosses above 30 from below, volume spike, above 1w EMA50
            long_cond = (rsi[i] > 30) and (rsi[i-1] <= 30) and volume_filter[i] and (close[i] > ema50_1w_aligned[i])
            # Short conditions: RSI crosses below 70 from above, volume spike, below 1w EMA50
            short_cond = (rsi[i] < 70) and (rsi[i-1] >= 70) and volume_filter[i] and (close[i] < ema50_1w_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI crosses back below 50 (neutral)
            if rsi[i] < 50 and rsi[i-1] >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI crosses back above 50 (neutral)
            if rsi[i] > 50 and rsi[i-1] <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals