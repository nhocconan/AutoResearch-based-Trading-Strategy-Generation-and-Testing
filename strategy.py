#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h trend-following with 4h momentum filter and volatility breakout
# Enter long when: 4h RSI > 55, price breaks above 1h Keltner upper band, volume > 1.3x average
# Enter short when: 4h RSI < 45, price breaks below 1h Keltner lower band, volume > 1.3x average
# Uses 4h RSI to filter trend direction, targeting 80-120 trades over 4 years
# Keltner channels adapt to volatility, reducing false breakouts in low volatility

name = "1h_keltner_breakout_4hrsi_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h RSI for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    delta_4h = pd.Series(close_4h).diff()
    gain_4h = delta_4h.clip(lower=0)
    loss_4h = -delta_4h.clip(upper=0)
    avg_gain_4h = gain_4h.ewm(alpha=1/14, adjust=False).mean()
    avg_loss_4h = loss_4h.ewm(alpha=1/14, adjust=False).mean()
    rs_4h = avg_gain_4h / (avg_loss_4h + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs_4h))
    rsi_4h = rsi_4h.values
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # 1h Keltner Channel (20 EMA, 2x ATR)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False).mean().values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    kc_upper = ema_20 + 2 * atr
    kc_lower = ema_20 - 2 * atr
    
    # Volume filter: volume > 1.3x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.3 * volume_ma
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(ema_20[i]) or 
            np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 1:  # long position
            # Exit: price below EMA(20) OR RSI < 45
            if close[i] < ema_20[i] or rsi_4h_aligned[i] < 45:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price above EMA(20) OR RSI > 55
            if close[i] > ema_20[i] or rsi_4h_aligned[i] > 55:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: volatility breakout + 4h RSI filter + volume + session
            if in_session and volume[i] > volume_threshold[i]:
                if rsi_4h_aligned[i] > 55 and close[i] > kc_upper[i]:
                    # Bullish breakout with 4h uptrend
                    signals[i] = 0.20
                    position = 1
                elif rsi_4h_aligned[i] < 45 and close[i] < kc_lower[i]:
                    # Bearish breakout with 4h downtrend
                    signals[i] = -0.20
                    position = -1
    
    return signals