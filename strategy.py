#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian Channel Breakout with 1w ATR regime filter and volume confirmation
# - Primary: 1d timeframe for lower frequency and reduced fee drag
# - HTF: 1w for volatility regime (ATR > 50th percentile) and trend filter (price > SMA50)
# - Long: Price breaks above Donchian(20) upper band + 1w ATR > 50th percentile + price > 1w SMA50
# - Short: Price breaks below Donchian(20) lower band + 1w ATR > 50th percentile + price < 1w SMA50
# - Exit: Price reverts to Donchian(20) midpoint or breaks opposite band (H/L)
# - Position sizing: 0.25 (discrete level)
# - Target: 50-100 total trades over 4 years (12-25/year) - within 1d sweet spot
# - Works in bull/bear: Donchian breakouts capture trends, ATR regime filter avoids low-vol chop

name = "1d_1w_donchian_atr_volume_v3"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
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
    
    # Calculate 1d Donchian Channel (20-period)
    highest_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_20 + lowest_20) / 2.0
    
    # Calculate 1w ATR(14) for volatility regime filter
    tr1 = pd.Series(high_1w).shift(1) - pd.Series(low_1w).shift(1)
    tr2 = abs(pd.Series(high_1w) - pd.Series(close_1w).shift(1))
    tr3 = abs(pd.Series(low_1w) - pd.Series(close_1w).shift(1))
    tr_1w = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = tr_1w.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1w ATR percentile rank (using 30-week lookback)
    atr_percentile = pd.Series(atr_1w).rolling(window=30, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1w, atr_percentile)
    
    # Calculate 1w SMA(50) for trend filter
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(sma_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # 1w volatility regime: ATR > 50th percentile (avoid low-vol chop)
        vol_regime = atr_percentile_aligned[i] > 50
        
        # 1w trend filter: price above/below 50-period SMA
        uptrend = close_1d[i] > sma_50_1w_aligned[i]
        downtrend = close_1d[i] < sma_50_1w_aligned[i]
        
        # Volume confirmation: current 1d volume > 1.2x 20-period MA
        volume_spike = volume_1d[i] > 1.2 * volume_ma_20_1d[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian upper + vol regime + uptrend + volume spike
            if (close_1d[i] > highest_20[i] and vol_regime and uptrend and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian lower + vol regime + downtrend + volume spike
            elif (close_1d[i] < lowest_20[i] and vol_regime and downtrend and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price reverts to Donchian midpoint (mean reversion)
            # 2. Price breaks opposite Donchian band (take profit)
            
            if position == 1:  # Long position
                exit_condition = (
                    close_1d[i] < donchian_mid[i] or  # Reverted to midpoint
                    close_1d[i] < lowest_20[i]        # Break below lower band (stop loss)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    close_1d[i] > donchian_mid[i] or  # Reverted to midpoint
                    close_1d[i] > highest_20[i]       # Break above upper band (stop loss)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals