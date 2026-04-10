#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation
# - Primary: 4h timeframe (proven sweet spot for trade frequency and Sharpe)
# - HTF: 1d for ATR percentile (volatility regime) and volume confirmation
# - Long: Price breaks above Donchian upper band (20-period high) + 1d ATR > 50th percentile + volume > 1.2x 20-period MA
# - Short: Price breaks below Donchian lower band (20-period low) + 1d ATR > 50th percentile + volume > 1.2x 20-period MA
# - Exit: Price reverts to Donchian midpoint (mean reversion) or ATR stoploss (2*ATR from entry)
# - Position sizing: 0.25 (discrete level)
# - Target: 75-200 total trades over 4 years (19-50/year) - within 4h sweet spot
# - Works in bull/bear: Donchian breakouts capture trends, ATR regime filter avoids low-vol chop, volume confirmation ensures participation

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
    # Upper band: 20-period high
    # Lower band: 20-period low
    # Middle band: (upper + lower) / 2
    high_roll = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    upper_band = high_roll
    lower_band = low_roll
    middle_band = (upper_band + lower_band) / 2.0
    
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
    entry_price = 0.0  # Track entry price for ATR stoploss
    atr_value = 0.0    # Track ATR value at entry for stoploss calculation
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # 1d volatility regime: ATR > 50th percentile (avoid low-vol chop)
        vol_regime = atr_percentile_aligned[i] > 50
        
        # Volume confirmation: current 1d volume > 1.2x 20-period MA
        volume_spike = volume_1d[i] > 1.2 * volume_ma_20_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above upper band + vol regime + volume spike
            if (close_4h[i] > upper_band[i] and vol_regime and volume_spike):
                position = 1
                entry_price = close_4h[i]
                # Calculate ATR-based stoploss distance (2 * ATR)
                # Need to get current ATR value - align 1d ATR to 4h
                atr_1d_series = pd.Series(atr_1d)
                atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_series.values)
                atr_value = atr_1d_aligned[i]
                signals[i] = 0.25
            # Short entry: Price breaks below lower band + vol regime + volume spike
            elif (close_4h[i] < lower_band[i] and vol_regime and volume_spike):
                position = -1
                entry_price = close_4h[i]
                # Calculate ATR-based stoploss distance (2 * ATR)
                atr_1d_series = pd.Series(atr_1d)
                atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_series.values)
                atr_value = atr_1d_aligned[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price reverts to middle band (mean reversion)
            # 2. ATR stoploss: 2 * ATR from entry price
            
            if position == 1:  # Long position
                # Calculate dynamic stoploss level
                stoploss_level = entry_price - 2.0 * atr_value
                
                exit_condition = (
                    close_4h[i] < middle_band[i] or  # Reverted to middle band (mean reversion)
                    close_4h[i] < stoploss_level     # ATR stoploss hit
                )
                if exit_condition:
                    position = 0
                    entry_price = 0.0
                    atr_value = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                # Calculate dynamic stoploss level
                stoploss_level = entry_price + 2.0 * atr_value
                
                exit_condition = (
                    close_4h[i] > middle_band[i] or  # Reverted to middle band (mean reversion)
                    close_4h[i] > stoploss_level     # ATR stoploss hit
                )
                if exit_condition:
                    position = 0
                    entry_price = 0.0
                    atr_value = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals