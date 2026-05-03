#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian(20) high with volume > 2x 20-bar average and close > 1w EMA50 (uptrend)
# Short when price breaks below Donchian(20) low with volume > 2x 20-bar average and close < 1w EMA50 (downtrend)
# Exit when price crosses 1d EMA10 (trailing exit) or trend fails (close crosses 1w EMA50)
# Donchian channels provide clear breakout levels, 1w EMA50 filters for higher-timeframe trend,
# volume confirmation reduces false breakouts. Works in bull (buy breakouts) and bear (sell breakdowns).
# Target: 30-100 total trades over 4 years = 7-25/year. Uses discrete sizing (0.25) to minimize fee churn.

name = "1d_Donchian20_Volume_1wEMA50_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Calculate 1d EMA10 for trailing exit
    close_s = pd.Series(close)
    ema_10 = close_s.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(50, 20, 20, 10) + 1  # EMA50(1w) + Donchian(20) + volume MA(20) + EMA10
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_spike[i]) or np.isnan(ema_10[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian(20) high with volume spike and close > 1w EMA50 (uptrend)
            if (close[i] > donchian_high[i-1] and  # Break above previous period's high
                volume_spike[i] and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below Donchian(20) low with volume spike and close < 1w EMA50 (downtrend)
            elif (close[i] < donchian_low[i-1] and  # Break below previous period's low
                  volume_spike[i] and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price crosses below 1d EMA10 or close < 1w EMA50 (trend failure)
            if (close[i] < ema_10[i] or 
                close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price crosses above 1d EMA10 or close > 1w EMA50 (trend failure)
            if (close[i] > ema_10[i] or 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals