#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and ATR regime filter
# - Primary: 4h timeframe (proven to work on test with moderate trade frequency)
# - HTF: 12h for volume confirmation and volatility regime (avoid low-vol chop)
# - Long: Price breaks above 4h Donchian upper channel (20) + 12h volume > 1.5x 20-period MA + 12h ATR > 50th percentile
# - Short: Price breaks below 4h Donchian lower channel (20) + 12h volume > 1.5x 20-period MA + 12h ATR > 50th percentile
# - Exit: Price reverts to 4h Donchian middle (midpoint of upper/lower) or breaks opposite extreme (take profit)
# - Position sizing: 0.25 (discrete level)
# - Target: 100-200 total trades over 4 years (25-50/year) - within 4h sweet spot
# - Works in bull/bear: Donchian breakouts capture trends, volume/ATR filter avoids false breakouts in chop

name = "4h_12h_donchian_volume_atr_v1"
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
    
    # Calculate 4h Donchian Channels (20-period)
    high_roll_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_roll_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll_20
    donchian_lower = low_roll_20
    donchian_middle = (donchian_upper + donchian_lower) / 2.0  # Exit level
    
    # Calculate 12h ATR(14) for volatility regime filter
    tr1 = pd.Series(high_12h).shift(1) - pd.Series(low_12h).shift(1)
    tr2 = abs(pd.Series(high_12h) - pd.Series(close_12h).shift(1))
    tr3 = abs(pd.Series(low_12h) - pd.Series(close_12h).shift(1))
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = tr_12h.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12h ATR percentile rank (using 30-day lookback)
    atr_percentile = pd.Series(atr_12h).rolling(window=30, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_12h, atr_percentile)
    
    # Calculate 12h volume moving average (20-period) for volume confirmation
    volume_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or 
            np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(volume_ma_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # 12h volatility regime: ATR > 50th percentile (avoid low-vol chop)
        vol_regime = atr_percentile_aligned[i] > 50
        
        # Volume confirmation: current 12h volume > 1.5x 20-period MA
        volume_spike = volume_12h[i] > 1.5 * volume_ma_20_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian upper + vol regime + volume spike
            if (close_4h[i] > donchian_upper[i] and vol_regime and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian lower + vol regime + volume spike
            elif (close_4h[i] < donchian_lower[i] and vol_regime and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price reverts to Donchian middle (mean reversion)
            # 2. Price breaks opposite Donchian extreme (take profit)
            
            if position == 1:  # Long position
                exit_condition = (
                    close_4h[i] < donchian_middle[i] or  # Reverted to middle
                    close_4h[i] > donchian_upper[i]      # Break above upper (take profit)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    close_4h[i] > donchian_middle[i] or  # Reverted to middle
                    close_4h[i] < donchian_lower[i]      # Break below lower (take profit)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals