#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d CCI + RSI mean reversion with weekly trend filter
# Uses weekly EMA50 to determine primary trend direction.
# CCI(20) < -100 indicates oversold in uptrend for long entries.
# CCI(20) > 100 indicates overbought in downtrend for short entries.
# RSI(14) confirms momentum divergence (RSI < 30 for long, > 70 for short).
# Volume spike filter ensures institutional participation.
# Designed for 1d timeframe to capture multi-day swings with low frequency.
# Target: 10-20 trades/year per symbol (40-80 total) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 50-period EMA on weekly close for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate CCI(20) on daily data
    typical_price = (high + low + close) / 3
    sma_tp = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = (typical_price - sma_tp) / (0.015 * mad)
    
    # Calculate RSI(14) on daily close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # Align indicators to daily timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(cci[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Uptrend (price > weekly EMA50) + CCI oversold (< -100) + RSI < 30 + volume spike
            if (close[i] > ema_50_1w_aligned[i] and 
                cci[i] < -100 and 
                rsi[i] < 30 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Downtrend (price < weekly EMA50) + CCI overbought (> 100) + RSI > 70 + volume spike
            elif (close[i] < ema_50_1w_aligned[i] and 
                  cci[i] > 100 and 
                  rsi[i] > 70 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: CCI returns to neutral zone or trend changes
            if position == 1:
                if (cci[i] > -50 or close[i] < ema_50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (cci[i] < 50 or close[i] > ema_50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_CCI_RSI_MeanReversion_WeeklyTrend"
timeframe = "1d"
leverage = 1.0