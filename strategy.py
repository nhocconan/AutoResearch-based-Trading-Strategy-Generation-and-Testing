#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h/1d confluence and volume confirmation
# - Primary: 6h timeframe for balance of signal frequency and fee drag
# - HTF: 12h for trend direction (Donchian breakout confirmation), 1d for volume/ATR regime
# - Long: Price breaks above 6h Donchian H20 + 12h close > 12h Donchian H20 (trend) + 1d volume > 1.5x 20-day MA + 1d ATR > 30th percentile
# - Short: Price breaks below 6h Donchian L20 + 12h close < 12h Donchian L20 (trend) + 1d volume > 1.5x 20-day MA + 1d ATR > 30th percentile
# - Exit: Price reverts to 6h Donchian midpoint (mean reversion) or breaks opposite H20/L20 with volume
# - Position sizing: 0.25 (discrete level)
# - Target: 50-150 total trades over 4 years (12-37/year) - within 6h sweet spot
# - Works in bull/bear: Donchian breakouts capture trends, volume/ATR filter avoids false breakouts in chop

name = "6h_12h_1d_donchian_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 6h Donchian Channels (20-period)
    high_ma_20_6h = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    low_ma_20_6h = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    donchian_mid_6h = (high_ma_20_6h + low_ma_20_6h) / 2.0
    
    # Calculate 12h Donchian Channels (20-period) for trend filter
    high_ma_20_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_ma_20_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align 12h Donchian levels to 6h bars
    high_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, high_ma_20_12h)
    low_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, low_ma_20_12h)
    close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
    
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
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(high_ma_20_6h[i]) or np.isnan(low_ma_20_6h[i]) or 
            np.isnan(high_ma_20_12h_aligned[i]) or np.isnan(low_ma_20_12h_aligned[i]) or
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
            # Long entry: Price breaks above 6h Donchian H20 + 12h trend up + vol regime + volume spike
            if (close_6h[i] > high_ma_20_6h[i] and 
                close_12h_aligned[i] > high_ma_20_12h_aligned[i] and
                vol_regime and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below 6h Donchian L20 + 12h trend down + vol regime + volume spike
            elif (close_6h[i] < low_ma_20_6h[i] and 
                  close_12h_aligned[i] < low_ma_20_12h_aligned[i] and
                  vol_regime and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price reverts to 6h Donchian midpoint (mean reversion)
            # 2. Price breaks opposite Donchian level with volume (stop and reverse)
            
            if position == 1:  # Long position
                exit_condition = (
                    close_6h[i] < donchian_mid_6h[i] or  # Reverted to midpoint
                    (close_6h[i] > high_ma_20_6h[i] and volume_1d[i] > 2.0 * volume_ma_20_1d_aligned[i])  # Strong break above
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    close_6h[i] > donchian_mid_6h[i] or  # Reverted to midpoint
                    (close_6h[i] < low_ma_20_6h[i] and volume_1d[i] > 2.0 * volume_ma_20_1d_aligned[i])  # Strong break below
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals