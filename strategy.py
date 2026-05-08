# 12h_Combined_Strategy_v1
# Hypothesis: Combine Donchian breakout with RSI mean reversion and volume confirmation on 12h timeframe.
# Long when price breaks above Donchian(20) high AND RSI < 30 (oversold) AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian(20) low AND RSI > 70 (overbought) AND volume > 1.5x 20-period average.
# Exit when price crosses back inside the Donchian channel.
# Uses volume confirmation to avoid false breakouts and RSI to avoid buying into strength or selling into weakness.
# Designed to work in both trending and ranging markets by filtering breakouts with momentum extremes.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled frequency to avoid fee drag.

name = "12h_Combined_Strategy_v1"
timeframe = "12h"
leverage = 1.0

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
    
    # Donchian(20) on 12h data
    donchian_period = 20
    upper_dc = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_dc = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # RSI(14) for momentum extreme
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi[np.isnan(rsi)] = 50  # Neutral value for warmup period
    
    # RSI conditions: oversold (<30) for long, overbought (>70) for short
    rsi_oversold = rsi < 30
    rsi_overbought = rsi > 70
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donchian_period, 20)  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_dc[i]) or np.isnan(lower_dc[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(rsi_oversold[i]) or np.isnan(rsi_overbought[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper, oversold RSI, high volume
            long_cond = (close[i] > upper_dc[i]) and rsi_oversold[i] and volume_filter[i]
            # Short conditions: price breaks below Donchian lower, overbought RSI, high volume
            short_cond = (close[i] < lower_dc[i]) and rsi_overbought[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Donchian lower
            if close[i] < lower_dc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Donchian upper
            if close[i] > upper_dc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals