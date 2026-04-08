# 12h_rsi_mean_reversion_1w_trend_volume_v1
# Hypothesis: On 12h timeframe, mean revert from RSI extremes when weekly trend supports it.
# Go long when RSI < 30 and price above weekly EMA50, short when RSI > 70 and price below weekly EMA50.
# Require volume > 20-period MA for confirmation. Weekly trend filter ensures we trade with higher timeframe momentum.
# Mean reversion works in both bull and bear markets as RSI extremes occur in all regimes.
# Target: 20-40 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_rsi_mean_reversion_1w_trend_volume_v1"
timeframe = "12h"
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
    
    # RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(rsi[i]) or np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI returns to neutral (50) or weekly trend turns against us
            if (rsi[i] >= 50) or (close[i] < ema_50_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral (50) or weekly trend turns against us
            if (rsi[i] <= 50) or (close[i] > ema_50_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: RSI oversold with volume confirmation AND weekly uptrend
            if (rsi[i] < 30) and (volume[i] > vol_ma[i]) and (close[i] > ema_50_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: RSI overbought with volume confirmation AND weekly downtrend
            elif (rsi[i] > 70) and (volume[i] > vol_ma[i]) and (close[i] < ema_50_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals