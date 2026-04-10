#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation
# - Primary: 4h timeframe for optimal trade frequency and proven performance
# - HTF: 1d for volatility regime (ATR percentile) and volume spike confirmation
# - Long: Price breaks above 4h Donchian(20) high + 1d ATR > 30th percentile + volume > 1.5x 20-period MA
# - Short: Price breaks below 4h Donchian(20) low + 1d ATR > 30th percentile + volume > 1.5x 20-period MA
# - Exit: Price reverts to 4h Donchian(20) midpoint or ATR-based trailing stop
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Target: 100-180 total trades over 4 years (25-45/year) - within 4h sweet spot
# - Works in bull/bear: ATR regime avoids low-vol chop, volume confirmation filters false breakouts

name = "4h_1d_donchian_atr_volume_v2"
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
    high_roll_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_roll_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high = high_roll_20
    donchian_low = low_roll_20
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
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
    entry_price = 0.0
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # 1d volatility regime: ATR > 30th percentile (avoid extreme low-vol chop)
        vol_regime = atr_percentile_aligned[i] > 30
        
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
            # 2. ATR-based trailing stop (2.5 * ATR from entry)
            
            if position == 1:  # Long position
                # Calculate ATR-based stop using 1d ATR scaled to 4h
                atr_1d_current = atr_1d[-1] if len(atr_1d) > 0 else 0.0  # Simplified for now
                # Use 4h ATR approximation: 1d ATR / 6 (approximate 6x 4h bars in 1d)
                atr_4h_approx = atr_1d_current / 6.0 if atr_1d_current > 0 else 0.0
                stop_price = entry_price - 2.5 * atr_4h_approx
                
                exit_condition = (
                    close_4h[i] < donchian_mid[i] or  # Reverted to midpoint
                    close_4h[i] < stop_price          # ATR trailing stop hit
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                # Calculate ATR-based stop using 1d ATR scaled to 4h
                atr_1d_current = atr_1d[-1] if len(atr_1d) > 0 else 0.0
                atr_4h_approx = atr_1d_current / 6.0 if atr_1d_current > 0 else 0.0
                stop_price = entry_price + 2.5 * atr_4h_approx
                
                exit_condition = (
                    close_4h[i] > donchian_mid[i] or  # Reverted to midpoint
                    close_4h[i] > stop_price          # ATR trailing stop hit
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals