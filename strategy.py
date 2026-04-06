#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1-week EMA trend filter and volume confirmation
# Enter long when: price breaks above Donchian(20) high AND price > 1w EMA(20) AND volume > 1.5x average
# Enter short when: price breaks below Donchian(20) low AND price < 1w EMA(20) AND volume > 1.5x average
# Exit when: price crosses opposite Donchian band OR RSI(14) reaches extreme (70/30) for mean reversion
# Target: 50-150 total trades over 4 years (12-37/year) with focus on major trend moves
# Works in bull markets via breakouts, in bear via short breakdowns, avoids chop via volume filter

name = "12h_donchian20_1wema_vol_v1"
timeframe = "12h"
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
    
    # Donchian channels (20-period) on 12h
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1-week EMA(20) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    # RSI(14) for mean reversion exit
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(volume_threshold[i]) or
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price < Donchian low OR RSI > 70 (overbought)
            if close[i] < low_min[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price > Donchian high OR RSI < 30 (oversold)
            if close[i] > high_max[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + trend filter + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > high_max[i] and close[i] > ema_20_aligned[i]:
                    # Breakout above upper band with weekly uptrend
                    signals[i] = 0.25
                    position = 1
                elif close[i] < low_min[i] and close[i] < ema_20_aligned[i]:
                    # Breakdown below lower band with weekly downtrend
                    signals[i] = -0.25
                    position = -1
    
    return signals