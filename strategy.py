#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ATR regime and volume confirmation
# - Primary: 6h timeframe targets 12-37 trades/year (50-150 total over 4 years)
# - HTF: 1d for ATR-based volatility regime and volume spike confirmation
# - Long: Price breaks above 20-period Donchian high + 1d ATR > 30th percentile + volume > 1.5x 20-period MA
# - Short: Price breaks below 20-period Donchian low + 1d ATR > 30th percentile + volume > 1.5x 20-period MA
# - Exit: Time-based exit after 3 bars (18 hours) or opposite Donchian break
# - Position sizing: 0.25 (discrete level)
# - Works in bull/bear: Donchian breakouts capture trends; ATR regime avoids low-vol chop; volume confirms conviction
# - Novelty: Combines Donchian structure with 1d volatility regime and volume spike (not recently tried on 6h)

name = "6h_1d_donchian_volume_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 6h Donchian channels (20-period)
    high_roll = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ATR(14) for volatility regime
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR percentile rank (30-day lookback)
    atr_percentile = pd.Series(atr_1d).rolling(window=30, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h timeframe
    high_roll_aligned = align_htf_to_ltf(prices, df_1d, high_roll)
    low_roll_aligned = align_htf_to_ltf(prices, df_1d, low_roll)
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    bars_since_entry = 0  # Track holding period for time-based exit
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(high_roll_aligned[i]) or np.isnan(low_roll_aligned[i]) or 
            np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        # Regime conditions
        # 1d volatility regime: ATR > 30th percentile (avoid extreme low-vol chop)
        vol_regime = atr_percentile_aligned[i] > 30
        
        # Volume confirmation: current 1d volume > 1.5x 20-period MA
        volume_spike = volume_1d[i] > 1.5 * volume_ma_20_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            bars_since_entry = 0
            # Long entry: Price breaks above Donchian high + vol regime + volume spike
            if (close_6h[i] > high_roll_aligned[i] and vol_regime and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low + vol regime + volume spike
            elif (close_6h[i] < low_roll_aligned[i] and vol_regime and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            bars_since_entry += 1
            # Exit conditions:
            # 1. Time-based exit: 3 bars (18 hours) to avoid overtrading
            # 2. Opposite Donchian break (reversal signal)
            
            time_exit = bars_since_entry >= 3
            
            if position == 1:  # Long position
                opposite_break = close_6h[i] < low_roll_aligned[i]
                if time_exit or opposite_break:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                opposite_break = close_6h[i] > high_roll_aligned[i]
                if time_exit or opposite_break:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals