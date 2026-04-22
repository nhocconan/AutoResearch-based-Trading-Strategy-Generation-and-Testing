# 1d_RSI_Overbought_Oversold_Volume_Confirmation_v1
# Hypothesis: Uses daily RSI extremes (>70 or <30) combined with volume spikes (>2x 20-period average) to identify exhaustion points in overbought/oversold conditions. 
# Works in both bull and bear markets by fading extremes with volume confirmation to avoid false signals. 
# Timeframe: 1d (daily bars) for lower frequency and higher quality signals.
# Expected trade frequency: 10-25 per year per symbol to minimize fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate RSI(14) on daily closes
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate weekly EMA(50) for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike filter: >2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # Align weekly EMA to daily timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) + price above weekly EMA50 + volume spike
            if (rsi[i] < 30 and 
                close[i] > ema50_1w_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) + price below weekly EMA50 + volume spike
            elif (rsi[i] > 70 and 
                  close[i] < ema50_1w_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: RSI returns to neutral zone (40-60) or trend reversal
            if position == 1:
                if (rsi[i] > 40 or 
                    close[i] < ema50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (rsi[i] < 60 or 
                    close[i] > ema50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_RSI_Overbought_Oversold_Volume_Confirmation_v1"
timeframe = "1d"
leverage = 1.0