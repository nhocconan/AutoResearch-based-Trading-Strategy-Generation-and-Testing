#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout with 12h ATR regime and volume confirmation
# - Primary: 4h timeframe for optimal trade frequency (target 20-50/year)
# - HTF: 12h for ATR percentile (volatility regime filter)
# - Long: Price breaks above Donchian(20) high + 12h ATR > 50th percentile + volume > 1.5x 20-period MA
# - Short: Price breaks below Donchian(20) low + 12h ATR > 50th percentile + volume > 1.5x 20-period MA
# - Exit: ATR-based trailing stop (3*ATR from extreme) or Donchian(10) opposite break
# - Position sizing: 0.25 (discrete level)
# - Works in bull/bear: Donchian captures breakouts in trending markets, ATR filter avoids low-vol chop, volume confirms conviction
# - Target: 75-200 total trades over 4 years (19-50/year) - within 4h sweet spot

name = "4h_12h_donchian_atr_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h OHLCV
    open_4h = prices['open'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    volume_4h = prices['volume'].values
    
    # Pre-compute 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 4h Donchian channels (20-period for entry, 10-period for exit)
    high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    high_10 = pd.Series(high_4h).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(low_4h).rolling(window=10, min_periods=10).min().values
    
    # Calculate 12h ATR(14) for volatility regime filter
    tr1 = pd.Series(high_12h).shift(1) - pd.Series(low_12h).shift(1)
    tr2 = abs(pd.Series(high_12h) - pd.Series(close_12h).shift(1))
    tr3 = abs(pd.Series(low_12h) - pd.Series(close_12h).shift(1))
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = tr_12h.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12h ATR percentile rank (using 50-bar lookback for stability)
    atr_percentile = pd.Series(atr_12h).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_12h, atr_percentile)
    
    # Calculate 12h volume moving average (20-period) for volume confirmation
    volume_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    long_extreme = 0.0  # highest high since entering long
    short_extreme = 0.0  # lowest low since entering short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(volume_ma_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # 12h volatility regime: ATR > 50th percentile (avoid low-vol chop)
        vol_regime = atr_percentile_aligned[i] > 50
        
        # Volume confirmation: current 12h volume > 1.5x 20-period MA
        volume_spike = volume_12h[i // 3] > 1.5 * volume_ma_20_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian(20) high + vol regime + volume spike
            if (close_4h[i] > high_20[i] and vol_regime and volume_spike):
                position = 1
                long_extreme = high_4h[i]
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian(20) low + vol regime + volume spike
            elif (close_4h[i] < low_20[i] and vol_regime and volume_spike):
                position = -1
                short_extreme = low_4h[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Update extremes
            if position == 1:  # Long position
                long_extreme = max(long_extreme, high_4h[i])
            else:  # Short position
                short_extreme = min(short_extreme, low_4h[i])
            
            # Calculate ATR-based trailing stop (using 4h ATR)
            tr1_4h = pd.Series(high_4h).shift(1) - pd.Series(low_4h).shift(1)
            tr2_4h = abs(pd.Series(high_4h) - pd.Series(close_4h).shift(1))
            tr3_4h = abs(pd.Series(low_4h) - pd.Series(close_4h).shift(1))
            tr_4h = pd.concat([tr1_4h, tr2_4h, tr3_4h], axis=1).max(axis=1)
            atr_4h = tr_4h.rolling(window=14, min_periods=14).mean().values
            
            # Exit conditions:
            # 1. ATR trailing stop (3*ATR from extreme)
            # 2. Donchian(10) opposite break (take profit)
            
            if position == 1:  # Long position
                atr_stop = long_extreme - 3.0 * atr_4h[i] if not np.isnan(atr_4h[i]) else low_4h[i]
                donchian_exit = close_4h[i] < low_10[i]
                exit_condition = (close_4h[i] < atr_stop) or donchian_exit
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                atr_stop = short_extreme + 3.0 * atr_4h[i] if not np.isnan(atr_4h[i]) else high_4h[i]
                donchian_exit = close_4h[i] > high_10[i]
                exit_condition = (close_4h[i] > atr_stop) or donchian_exit
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals