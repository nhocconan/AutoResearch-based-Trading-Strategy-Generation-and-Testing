#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h trend filter (EMA34) and 1h RSI mean reversion.
# Uses 4h EMA34 for trend direction and 1h RSI(14) for entry timing.
# Enters long when trend is up (price > 4h EMA34) and RSI < 30 (oversold).
# Enters short when trend is down (price < 4h EMA34) and RSI > 70 (overbought).
# Includes volume confirmation (volume > 1.5x 20-period average) and session filter (08-20 UTC).
# Designed for low trade frequency (~15-25/year) to avoid fee drag.
name = "1h_4h_EMA34_RSI14_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA34 trend (called ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # RSI(14) on 1h data
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for RSI and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(volume_ma[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: uptrend (price > 4h EMA34) and oversold (RSI < 30) with volume
            if (close[i] > ema_34_4h_aligned[i] and 
                rsi_values[i] < 30 and 
                volume_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: downtrend (price < 4h EMA34) and overbought (RSI > 70) with volume
            elif (close[i] < ema_34_4h_aligned[i] and 
                  rsi_values[i] > 70 and 
                  volume_filter[i]):
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long: exit if trend changes (price < 4h EMA34) or RSI overbought (RSI > 70)
            if close[i] < ema_34_4h_aligned[i] or rsi_values[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short: exit if trend changes (price > 4h EMA34) or RSI oversold (RSI < 30)
            if close[i] > ema_34_4h_aligned[i] or rsi_values[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals