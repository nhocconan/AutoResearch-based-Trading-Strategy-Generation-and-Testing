#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + ATR regime filter
# - Primary: 4h timeframe for optimal trade frequency (target 75-200 total trades)
# - HTF: 1d for volume confirmation (spike vs 20-period MA) and volatility regime (ATR > 40th percentile)
# - Long: Price breaks above 20-period Donchian high + 1d volume > 1.5x 20-period MA + 1d ATR > 40th percentile
# - Short: Price breaks below 20-period Donchian low + 1d volume > 1.5x 20-period MA + 1d ATR > 40th percentile
# - Exit: Price reverts to 20-period Donchian midpoint (mean reversion) or ATR-based stoploss
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Stoploss: 2.0 * ATR(20) from entry price (trailing for longs/shorts)
# - Works in bull/bear: Donchian breakouts capture trends; volume/vol filters avoid false breakouts in chop

name = "4h_1d_donchian_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for Donchian(20) and ATR(20)
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
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0  # Mean reversion exit
    
    # Calculate 4h ATR(20) for stoploss
    tr1 = pd.Series(high_4h).shift(1) - pd.Series(low_4h).shift(1)
    tr2 = abs(pd.Series(high_4h) - pd.Series(close_4h).shift(1))
    tr3 = abs(pd.Series(low_4h) - pd.Series(close_4h).shift(1))
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h = tr_4h.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ATR(14) for volatility regime filter
    tr1_1d = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2_1d = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3_1d = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr_1d = pd.concat([tr1_1d, tr2_1d, tr3_1d], axis=1).max(axis=1)
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
    entry_price = 0.0  # Track entry price for stoploss
    
    for i in range(40, n):  # Start after warmup period (max of 20,20,14,30)
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_4h[i]) or np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # 1d volatility regime: ATR > 40th percentile (avoid low-vol chop)
        vol_regime = atr_percentile_aligned[i] > 40
        
        # Volume confirmation: current 1d volume > 1.5x 20-period MA
        volume_spike = volume_1d[i] > 1.5 * volume_ma_20_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian high + vol regime + volume spike
            if (close_4h[i] > donchian_high[i] and vol_regime and volume_spike):
                position = 1
                entry_price = close_4h[i]
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low + vol regime + volume spike
            elif (close_4h[i] < donchian_low[i] and vol_regime and volume_spike):
                position = -1
                entry_price = close_4h[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price reverts to Donchian midpoint (mean reversion)
            # 2. ATR-based stoploss (2.0 * ATR from entry price)
            
            if position == 1:  # Long position
                stoploss_level = entry_price - 2.0 * atr_4h[i]
                exit_condition = (
                    close_4h[i] < donchian_mid[i] or      # Reverted to midpoint
                    close_4h[i] < stoploss_level          # Stoploss hit
                )
                if exit_condition:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                stoploss_level = entry_price + 2.0 * atr_4h[i]
                exit_condition = (
                    close_4h[i] > donchian_mid[i] or      # Reverted to midpoint
                    close_4h[i] > stoploss_level          # Stoploss hit
                )
                if exit_condition:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals