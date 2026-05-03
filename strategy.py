#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Uses 1d timeframe for low trade frequency, with 1w for trend direction.
# Donchian levels from prior completed 1d bar provide clear breakout levels.
# Breakouts with volume indicate institutional participation. Trend filter avoids counter-trend trades.
# Discrete sizing 0.25 to manage drawdown. Target: 50-100 total trades over 4 years.

name = "1d_Donchian20_1wEMA50_VolumeSpike_Trend"
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA50 trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) for 1d data (for stoploss)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0
    lowest_low_since_entry = 0
    
    for i in range(100, n):
        # Skip if any value is NaN
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Calculate Donchian levels for prior completed 1d bar
        lookback_start = max(0, i - 20)
        lookback_end = i  # Exclude current bar
        if lookback_end - lookback_start < 20:
            # Not enough prior bars for Donchian calculation
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        prior_high = np.max(high[lookback_start:lookback_end])
        prior_low = np.min(low[lookback_start:lookback_end])
        
        # Volume confirmation: current 1d volume > 1.5x 20-period MA
        vol_ma_20 = np.mean(volume[max(0, i-20):i]) if i >= 20 else 0
        volume_spike = volume[i] > (1.5 * vol_ma_20) if vol_ma_20 > 0 else False
        
        # Entry conditions
        # Long: break above prior 20-period high with volume spike, above 1w EMA50
        long_entry = (close[i] > prior_high) and volume_spike and (close[i] > ema_50_1w_aligned[i])
        # Short: break below prior 20-period low with volume spike, below 1w EMA50
        short_entry = (close[i] < prior_low) and volume_spike and (close[i] < ema_50_1w_aligned[i])
        
        # Exit conditions (ATR-based trailing stop)
        long_exit = False
        short_exit = False
        
        if position == 1:  # Long position
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            long_exit = close[i] < (highest_high_since_entry - 2.5 * atr[i])
        elif position == -1:  # Short position
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            short_exit = close[i] > (lowest_low_since_entry + 2.5 * atr[i])
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            elif short_entry:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals