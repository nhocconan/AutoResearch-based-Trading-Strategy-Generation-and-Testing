#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h EMA(20) for trend and 1h RSI(14) pullback for entry.
# 4h EMA > 1h close for long bias, EMA < 1h close for short bias.
# Enter on RSI pullbacks: long when RSI < 30 in uptrend, short when RSI > 70 in downtrend.
# Volume confirmation: current volume > 1.5x 20-period average reduces false signals.
# Session filter: only trade 08-20 UTC to avoid low-liquidity hours.
# Fixed position size 0.20 to limit risk and control trade frequency.
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate EMA(20) on 4h close
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate RSI(14) on 1h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # Fixed 20% position
    
    # Start after enough data for calculations
    start = max(20, 14, 20)  # EMA, RSI, volume MA
    
    for i in range(start, n):
        # Skip if outside trading session or missing data
        if (not in_session[i] or 
            np.isnan(ema_4h_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: 4h EMA vs 1h close
        uptrend = close[i] > ema_4h_aligned[i]
        downtrend = close[i] < ema_4h_aligned[i]
        
        if position == 0:
            # Look for RSI pullback entries
            if uptrend and (rsi[i] < 30) and volume_confirmed:
                # Long on RSI oversold in uptrend
                position = 1
                signals[i] = position_size
            elif downtrend and (rsi[i] > 70) and volume_confirmed:
                # Short on RSI overbought in downtrend
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI reaches overbought or trend changes
            if (rsi[i] > 70) or (close[i] < ema_4h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI reaches oversold or trend changes
            if (rsi[i] < 30) or (close[i] > ema_4h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4hEMA_RSI_Pullback_VolumeFilter_v1"
timeframe = "1h"
leverage = 1.0