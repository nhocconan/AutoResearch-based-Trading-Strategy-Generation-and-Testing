#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR volatility filter and volume confirmation
# - Primary: 4h timeframe for optimal trade frequency (19-50/year)
# - HTF: 1d for ATR percentile regime filter (avoid low-vol chop) and volume spike confirmation
# - Long: Price breaks above 20-period Donchian high + 1d ATR > 30th percentile + volume > 1.5x 20-period MA
# - Short: Price breaks below 20-period Donchian low + 1d ATR > 30th percentile + volume > 1.5x 20-period MA
# - Exit: ATR-based trailing stop (3*ATR) or Donchian opposite breakout
# - Position sizing: 0.25 (discrete level)
# - Target: 80-180 total trades over 4 years (20-45/year) - within 4h sweet spot
# - Works in bull/bear: Donchian captures breakouts in trending markets, ATR filter avoids whipsaws in ranging markets

name = "4h_1d_donchian_atr_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h OHLCV
    open_4h = prices['open'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    volume_4h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 4h Donchian Channel (20-period)
    high_roll = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ATR(14) for volatility regime filter
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR percentile rank (using 30-day lookback)
    atr_percentile = pd.Series(atr_1d).rolling(window=30, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_long = 0.0  # Track highest close since long entry
    lowest_since_short = 0.0  # Track lowest close since short entry
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # 1d volatility regime: ATR > 30th percentile (avoid low-vol chop)
        vol_regime = atr_percentile_aligned[i] > 30
        
        # Volume confirmation: current 1d volume > 1.5x 20-period MA
        volume_spike = volume_1d[i] > 1.5 * volume_ma_20_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian high + vol regime + volume spike
            if (close_4h[i] > high_roll[i] and vol_regime and volume_spike):
                position = 1
                highest_since_long = close_4h[i]
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low + vol regime + volume spike
            elif (close_4h[i] < low_roll[i] and vol_regime and volume_spike):
                position = -1
                lowest_since_short = close_4h[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Update tracking values
            if position == 1:  # Long position
                highest_since_long = max(highest_since_long, close_4h[i])
                # Exit conditions:
                # 1. ATR trailing stop: price drops 3*ATR from highest since entry
                # 2. Donchian opposite breakout: price breaks below Donchian low
                atr_value = atr_1d[i // 16] if i // 16 < len(atr_1d) else atr_1d[-1]  # Approximate 1d ATR for 4h bar
                trailing_stop = highest_since_long - 3.0 * atr_value
                exit_condition = (
                    close_4h[i] < trailing_stop or  # ATR trailing stop hit
                    close_4h[i] < low_roll[i]       # Donchian opposite breakout
                )
                if exit_condition:
                    position = 0
                    highest_since_long = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                lowest_since_short = min(lowest_since_short, close_4h[i])
                # Exit conditions:
                # 1. ATR trailing stop: price rises 3*ATR from lowest since entry
                # 2. Donchian opposite breakout: price breaks above Donchian high
                atr_value = atr_1d[i // 16] if i // 16 < len(atr_1d) else atr_1d[-1]  # Approximate 1d ATR for 4h bar
                trailing_stop = lowest_since_short + 3.0 * atr_value
                exit_condition = (
                    close_4h[i] > trailing_stop or  # ATR trailing stop hit
                    close_4h[i] > high_roll[i]      # Donchian opposite breakout
                )
                if exit_condition:
                    position = 0
                    lowest_since_short = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals