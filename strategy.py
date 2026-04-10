#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla Pivot Breakout with 1w trend filter and volume confirmation
# - Primary: 6h timeframe for moderate frequency (target: 12-37 trades/year)
# - HTF: 1w for trend direction (price above/below weekly SMA50) and 1d for volatility/volume
# - Long: Price breaks above H3 Camarilla pivot (from 1d) + price > weekly SMA50 + 1d ATR > 30th percentile + volume > 1.5x 20-period MA
# - Short: Price breaks below L3 Camarilla pivot (from 1d) + price < weekly SMA50 + 1d ATR > 30th percentile + volume > 1.5x 20-period MA
# - Exit: Price reverts to Camarilla Pivot Point (mean reversion) or breaks H4/L4
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Weekly SMA50 filter ensures we only trade with the dominant trend, reducing whipsaws in ranging markets

name = "6h_1w_1d_camarilla_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 50 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 1w data
    close_1w = df_1w['close'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 6h Camarilla Pivot Points (based on previous 1d)
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Calculate Camarilla levels for each 6h bar (using previous day's OHLC)
    rng = high_1d_aligned - low_1d_aligned
    h3 = close_1d_aligned + 1.25 * rng  # Long entry: break above H3
    l3 = close_1d_aligned - 1.25 * rng  # Short entry: break below L3
    h4 = close_1d_aligned + 1.5 * rng   # Long exit: break above H4 (take profit)
    l4 = close_1d_aligned - 1.5 * rng   # Short exit: break below L4 (take profit)
    pivot = (high_1d_aligned + low_1d_aligned + close_1d_aligned) / 3.0  # Mean reversion exit
    
    # Calculate 1w SMA50 for trend filter
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
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
        if (np.isnan(h3[i]) or np.isnan(l3[i]) or 
            np.isnan(sma50_1w_aligned[i]) or 
            np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # Weekly trend filter: price above/below 50-period SMA
        uptrend = close_6h[i] > sma50_1w_aligned[i]
        downtrend = close_6h[i] < sma50_1w_aligned[i]
        
        # 1d volatility regime: ATR > 30th percentile (avoid low-vol chop)
        vol_regime = atr_percentile_aligned[i] > 30
        
        # Volume confirmation: current 1d volume > 1.5x 20-period MA
        volume_spike = volume_1d[i] > 1.5 * volume_ma_20_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above H3 resistance + uptrend + vol regime + volume spike
            if (close_6h[i] > h3[i] and uptrend and vol_regime and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below L3 support + downtrend + vol regime + volume spike
            elif (close_6h[i] < l3[i] and downtrend and vol_regime and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price reverts to Pivot Point (mean reversion)
            # 2. Price breaks opposite H4/L4 level (take profit)
            
            if position == 1:  # Long position
                exit_condition = (
                    close_6h[i] < pivot[i] or  # Reverted to pivot
                    close_6h[i] > h4[i]        # Break above H4 (take profit)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    close_6h[i] > pivot[i] or  # Reverted to pivot
                    close_6h[i] < l4[i]        # Break below L4 (take profit)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals