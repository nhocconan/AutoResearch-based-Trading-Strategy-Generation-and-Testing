#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA(50) trend filter and volume confirmation
# Enter long when: price breaks above Donchian(20) high, price > 12h EMA(50), volume > 1.5x avg
# Enter short when: price breaks below Donchian(20) low, price < 12h EMA(50), volume > 1.5x avg
# Exit when: price touches opposite Donchian band OR RSI(14) reaches extreme (>70 long, <30 short)
# Uses 12h trend to filter counter-trend breakouts, targeting 100-180 trades over 4 years

name = "4h_donchian20_12h_ema_vol_rsi_exit_v1"
timeframe = "4h"
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
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donch_high = high_roll.values
    donch_low = low_roll.values
    
    # 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    # RSI(14) for exit signals
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
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]) or
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price touches lower Donchian band OR RSI > 70 (overbought)
            if close[i] <= donch_low[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price touches upper Donchian band OR RSI < 30 (oversold)
            if close[i] >= donch_high[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries: Donchian breakout + trend filter + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > donch_high[i] and close[i] > ema_50_aligned[i]:
                    # Bullish breakout above Donchian high with 12h uptrend
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donch_low[i] and close[i] < ema_50_aligned[i]:
                    # Bearish breakout below Donchian low with 12h downtrend
                    signals[i] = -0.25
                    position = -1
    
    return signals