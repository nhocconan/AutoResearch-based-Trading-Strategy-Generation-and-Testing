#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume confirmation and ATR regime filter
# - Primary: 1d timeframe for lower frequency and reduced fee drag (target: 30-100 trades over 4 years)
# - HTF: 1w for volume confirmation (weekly volume > 1.5x 4-week MA) and ATR regime (weekly ATR > 50th percentile)
# - Long: Price breaks above Donchian(20) high + weekly volume confirmation + weekly ATR > 50th percentile
# - Short: Price breaks below Donchian(20) low + weekly volume confirmation + weekly ATR > 50th percentile
# - Exit: Price reverts to Donchian(20) midpoint (mean reversion) or breaks opposite Donchian(40) level
# - Position sizing: 0.25 (discrete level)
# - Works in bull/bear: Donchian breakouts capture trends, mean reversion exit works in ranging markets (2025)

name = "1d_1w_donchian_volume_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d OHLCV
    open_1d = prices['open'].values
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    volume_1d = prices['volume'].values
    
    # Pre-compute 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1d Donchian(20) channels
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2.0
    
    # Calculate 1d Donchian(40) for exit levels
    high_40 = pd.Series(high_1d).rolling(window=40, min_periods=40).max().values
    low_40 = pd.Series(low_1d).rolling(window=40, min_periods=40).min().values
    
    # Calculate 1w ATR(14) for volatility regime filter
    tr1 = pd.Series(high_1w).shift(1) - pd.Series(low_1w).shift(1)
    tr2 = abs(pd.Series(high_1w) - pd.Series(close_1w).shift(1))
    tr3 = abs(pd.Series(low_1w) - pd.Series(close_1w).shift(1))
    tr_1w = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = tr_1w.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1w ATR percentile rank (using 20-week lookback)
    atr_percentile = pd.Series(atr_1w).rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1w, atr_percentile)
    
    # Calculate 1w volume moving average (4-period) for volume confirmation
    volume_ma_4_1w = pd.Series(volume_1w).rolling(window=4, min_periods=4).mean().values
    volume_ma_4_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_4_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(40, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(high_40[i]) or np.isnan(low_40[i]) or
            np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(volume_ma_4_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # 1w volatility regime: ATR > 50th percentile (avoid low-vol chop)
        vol_regime = atr_percentile_aligned[i] > 50
        
        # Volume confirmation: current 1w volume > 1.5x 4-period MA
        volume_spike = volume_1w[i // 7] > 1.5 * volume_ma_4_1w_aligned[i] if i >= 7 else False
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian(20) high + vol regime + volume spike
            if (close_1d[i] > high_20[i] and vol_regime and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian(20) low + vol regime + volume spike
            elif (close_1d[i] < low_20[i] and vol_regime and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price reverts to Donchian(20) midpoint (mean reversion)
            # 2. Price breaks opposite Donchian(40) level (take profit/stop)
            
            if position == 1:  # Long position
                exit_condition = (
                    close_1d[i] < donchian_mid[i] or  # Reverted to midpoint
                    close_1d[i] > high_40[i]          # Break above Donchian(40) (take profit)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    close_1d[i] > donchian_mid[i] or  # Reverted to midpoint
                    close_1d[i] < low_40[i]           # Break below Donchian(40) (take profit)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals