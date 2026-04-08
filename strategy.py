# 4h_donchian_breakout_1d_trend_volume_v2
# Hypothesis: On 4h timeframe, use Donchian channel breakout with daily trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high with volume > 1.5x average and daily trend up.
# Short when price breaks below Donchian(20) low with volume > 1.5x average and daily trend down.
# Exit on opposite Donchian band touch or when volume drops below average.
# Daily trend defined by price above/below daily EMA50.
# This strategy targets ~25-50 trades/year by requiring multiple confirmations.
# Works in both bull and bear markets via trend filter and volatility-based exits.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v2"
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
    
    # Donchian channel (20-period) on 4h
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily trend filter: EMA50
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    daily_close = df_daily['close'].values
    daily_ema50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    daily_ema50_4h = align_htf_to_ltf(prices, df_daily, daily_ema50)
    
    # Volume confirmation: 20-period average on 4h
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(daily_ema50_4h[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price touches lower Donchian band or volume drops below average
            if close[i] <= low_roll[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches upper Donchian band or volume drops below average
            if close[i] >= high_roll[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Daily trend filter
            daily_uptrend = close[i] > daily_ema50_4h[i]
            daily_downtrend = close[i] < daily_ema50_4h[i]
            
            # Long entry: price breaks above upper Donchian band with volume and uptrend
            if close[i] > high_roll[i] and volume_ok and daily_uptrend:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower Donchian band with volume and downtrend
            elif close[i] < low_roll[i] and volume_ok and daily_downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals