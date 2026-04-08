#5min_rsi_volume_mean_reversion_v1
# Strategy: 5-minute RSI mean reversion with volume confirmation and 15m trend filter
# Hypothesis: In ranging markets, RSI extremes combined with volume spikes provide mean reversion opportunities.
# The 15m EMA50 acts as a trend filter to avoid counter-trend trades during strong moves.
# Volume confirmation ensures institutional participation.
# Target: 100-200 trades per year (balanced for 5m timeframe) to manage fee drag.
# Works in both bull and bear markets by fading extremes only when volume confirms reversal intent.

name = "5min_rsi_volume_mean_reversion_v1"
timeframe = "5m"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 15m EMA trend filter (50-period)
    df_15m = get_htf_data(prices, '15m')
    if len(df_15m) < 1:
        return np.zeros(n)
    
    ema_15m = pd.Series(df_15m['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_15m_aligned = align_htf_to_ltf(prices, df_15m, ema_15m)
    
    # RSI calculation (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: volume > 1.5x 20-period average
    vol_period = 20
    vol_ma = np.full(n, np.nan)
    vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    
    # Start from sufficient lookback
    start_idx = max(14, 20, 50) + 5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_15m_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: RSI crosses above 50 or trend fails
            if rsi[i] >= 50 or close[i] < ema_15m_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI crosses below 50 or trend fails
            if rsi[i] <= 50 or close[i] > ema_15m_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Mean reversion entries at RSI extremes with volume confirmation
            # Long when RSI < 30 (oversold) and above 15m EMA (bullish bias)
            if (rsi[i] < 30 and close[i] > ema_15m_aligned[i] and volume_filter):
                position = 1
                signals[i] = 0.25
            # Short when RSI > 70 (overbought) and below 15m EMA (bearish bias)
            elif (rsi[i] > 70 and close[i] < ema_15m_aligned[i] and volume_filter):
                position = -1
                signals[i] = -0.25
    
    return signals