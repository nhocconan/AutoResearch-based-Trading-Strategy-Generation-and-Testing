#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly trend filter (price > weekly EMA50) and volume confirmation.
# Uses 6h timeframe for signal generation with 1d/1w for trend/volume regime to control trade frequency.
# Donchian breakouts capture institutional order flow. Weekly EMA50 filter ensures alignment with higher timeframe trend.
# Volume confirmation (current volume > 1.5x 20-period MA) filters weak breakouts.
# Discrete sizing 0.25 to balance return and drawdown. Target: 50-150 total trades over 4 years.

name = "6h_Donchian20_1wEMA50_VolumeSpike_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly trend calculation (we need daily to build weekly)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 from daily data (approximate weekly by using 5-day EMA on daily)
    # Since we don't have direct weekly data, we use 5-period EMA on daily as proxy for weekly trend
    close_1d = df_1d['close'].values
    ema_5_1d = pd.Series(close_1d).ewm(span=5, min_periods=5, adjust=False).mean().values
    ema_5_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_5_1d)
    
    # Calculate 1d volume regime (high volume when current volume > 1.5x 20-period MA)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_regime_1d = vol_1d > (1.5 * vol_ma_1d)
    vol_regime_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_regime_1d)
    
    # Calculate Donchian channels (20-period) on 6h data
    high_rolling_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for 6h data (for stoploss)
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
        # Get current values
        donchian_high = high_rolling_max[i]
        donchian_low = low_rolling_min[i]
        weekly_ema_trend = ema_5_1d_aligned[i]
        vol_reg = vol_regime_1d_aligned[i]
        atr_val = atr[i]
        
        # Skip if any value is NaN
        if np.isnan(donchian_high) or np.isnan(donchian_low) or np.isnan(weekly_ema_trend) or np.isnan(vol_reg) or np.isnan(atr_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Volume confirmation: current 6h volume > 1.5x 20-period MA
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        volume_spike = volume[i] > (1.5 * vol_ma_20)
        
        # Entry conditions
        # Long: break above Donchian high with volume spike, above weekly EMA50
        long_entry = (close[i] > donchian_high) and volume_spike and (close[i] > weekly_ema_trend)
        # Short: break below Donchian low with volume spike, below weekly EMA50
        short_entry = (close[i] < donchian_low) and volume_spike and (close[i] < weekly_ema_trend)
        
        # Exit conditions (ATR-based trailing stop)
        long_exit = False
        short_exit = False
        
        if position == 1:  # Long position
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            long_exit = close[i] < (highest_high_since_entry - 2.5 * atr_val)
        elif position == -1:  # Short position
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            short_exit = close[i] > (lowest_low_since_entry + 2.5 * atr_val)
        
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