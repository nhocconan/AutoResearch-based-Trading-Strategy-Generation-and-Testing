#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter (EMA34 > EMA89 = uptrend) and volume confirmation
# - Long when price breaks above Donchian upper (20-bar high) in 1d uptrend with volume > 1.5x 20-period average
# - Short when price breaks below Donchian lower (20-bar low) in 1d downtrend with volume confirmation
# - Exit when price closes opposite Donchian band (lower for longs, upper for shorts)
# - Fixed position size 0.25 to control drawdown
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - Works in bull markets via breakouts, in bear markets via short breakdowns + trend filter avoids whipsaws

name = "4h_1d_donchian_breakout_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 89:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMAs for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89_1d = pd.Series(close_1d).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 1d EMAs to 4h timeframe (wait for completed 1d bar)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_89_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_89_1d)
    
    # Pre-compute Donchian channels (20-period) for 4h
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute volume confirmation (20-period average for 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_89_1d_aligned[i]) or
            np.isnan(high_ma_20[i]) or np.isnan(low_ma_20[i]) or
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: 1d EMA34 > EMA89 = uptrend, < = downtrend
        uptrend = ema_34_1d_aligned[i] > ema_89_1d_aligned[i]
        downtrend = ema_34_1d_aligned[i] < ema_89_1d_aligned[i]
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit when price closes below Donchian lower (20-bar low)
            if close[i] < low_ma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit when price closes above Donchian upper (20-bar high)
            if close[i] > high_ma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Breakout entry with volume confirmation and trend alignment
            if volume_confirmed:
                # Long: break above Donchian upper in uptrend
                if uptrend and close[i] > high_ma_20[i]:
                    position = 1
                    signals[i] = position_size
                # Short: break below Donchian lower in downtrend
                elif downtrend and close[i] < low_ma_20[i]:
                    position = -1
                    signals[i] = -position_size
    
    return signals