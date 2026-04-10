#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# - Donchian levels from 1d: upper/lower 20-day channel as breakout levels
# - Breakout above upper channel (long) or below lower channel (short) with volume confirmation
# - 1w EMA50 trend filter ensures we trade with higher timeframe trend (avoids counter-trend in bear markets)
# - Volume confirmation: current volume > 2.0x 20-period average to avoid false breakouts
# - Exit: touch of opposite Donchian level (lower for longs, upper for shorts)
# - Position size: 0.25 (25% of capital) for balanced risk/return
# - Target: 10-30 trades/year on 1d (40-120 total over 4 years) to minimize fee drag
# - Works in both bull/bear: EMA50 trend filter adapts to regime, volume confirmation reduces whipsaws

name = "1d_donchian_breakout_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Pre-compute 1d Donchian channels (20-period)
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    # Upper channel = highest high over past 20 periods
    upper_channel = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower channel = lowest low over past 20 periods
    lower_channel = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 1d volume average (20-period)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(trend_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirm = volume[i] > 2.0 * vol_ma_20[i]
        
        # 1w trend filter: price > EMA50 = bullish, price < EMA50 = bearish
        bullish_trend = not np.isnan(trend_aligned[i]) and close_1d[i] > trend_aligned[i]
        bearish_trend = not np.isnan(trend_aligned[i]) and close_1d[i] < trend_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price > upper_channel AND bullish trend AND volume confirmation
            if close_1d[i] > upper_channel[i] and bullish_trend and volume_confirm:
                position = 1
                signals[i] = 0.25
            # Short conditions: price < lower_channel AND bearish trend AND volume confirmation
            elif close_1d[i] < lower_channel[i] and bearish_trend and volume_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price touches opposite Donchian level
            exit_long = close_1d[i] < lower_channel[i]   # Price breaks below lower channel (exit long)
            exit_short = close_1d[i] > upper_channel[i]  # Price breaks above upper channel (exit short)
            
            if (position == 1 and exit_long) or (position == -1 and exit_short):
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals