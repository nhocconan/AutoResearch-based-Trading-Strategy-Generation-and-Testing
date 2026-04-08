#!/usr/bin/env python3
"""
1d EMA Trend with RSI Pullback and Volume Confirmation
Hypothesis: In trending markets, price pulls back to the EMA before continuing.
Buy near EMA during uptrend with RSI showing exhaustion (not oversold) and volume confirmation.
Sell/short near EMA during downtrend with RSI showing exhaustion (not overbought) and volume.
Uses 1w EMA filter to ensure alignment with higher timeframe trend.
Targets 15-25 trades/year on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_ema_trend_rsi_pullback_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w EMA(50) for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = df_1w['close'].ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d EMA(50) for trend
    ema_50 = close.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume filter (>1.3x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI > 70 (overbought) or price breaks below EMA
            if (rsi[i] > 70 or close[i] < ema_50[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 30 (oversold) or price breaks above EMA
            if (rsi[i] < 30 or close[i] > ema_50[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price near EMA from below, RSI recovering from oversold, uptrend, volume
            if (close[i] > ema_50[i] and  # Price above EMA
                close[i-1] <= ema_50[i-1] and  # Was at or below EMA yesterday
                rsi[i] > 30 and rsi[i] < 50 and  # RSI recovering but not overbought
                ema_50[i] > ema_50[i-1] and  # EMA rising (uptrend)
                ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1] and  # 1w EMA rising
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price near EMA from above, RSI declining from overbought, downtrend, volume
            elif (close[i] < ema_50[i] and  # Price below EMA
                  close[i-1] >= ema_50[i-1] and  # Was at or above EMA yesterday
                  rsi[i] < 70 and rsi[i] > 50 and  # RSI declining but not oversold
                  ema_50[i] < ema_50[i-1] and  # EMA falling (downtrend)
                  ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1] and  # 1w EMA falling
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals