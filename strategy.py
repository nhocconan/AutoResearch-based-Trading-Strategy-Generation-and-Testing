#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Volume Regime Filter
# - Primary: 6h timeframe for balanced frequency and reduced fee drag (target: 50-150 total trades over 4 years)
# - HTF: 1w for trend direction (EMA34 slope) and 1d for volatility regime (ATR percentile)
# - Long: Elder Bull Power > 0 + Bear Power < 0 + weekly EMA34 sloping up + 1d ATR > 50th percentile
# - Short: Elder Bull Power < 0 + Bear Power > 0 + weekly EMA34 sloping down + 1d ATR > 50th percentile
# - Exit: Elder Bull Power crosses below 0 (long) or above 0 (short) OR ATR drops below 30th percentile (volatility collapse)
# - Position sizing: 0.25 (discrete level)
# - Works in bull/bear: Elder Ray measures bull/bear power relative to EMA13; weekly EMA34 filter ensures we trade with higher timeframe trend; ATR regime avoids low-vol chop

name = "6h_1w_1d_elderray_volume_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 40 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 6h EMA13 for Elder Ray
    close_6h_series = pd.Series(close_6h)
    ema13_6h = close_6h_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Bull Power and Bear Power for 6h
    bull_power = high_6h - ema13_6h  # High - EMA13
    bear_power = low_6h - ema13_6h   # Low - EMA13
    
    # Calculate 1w EMA34 for trend direction
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1w EMA34 slope (trend direction) - using 3-bar lookback for stability
    ema34_slope = np.zeros_like(ema34_1w)
    ema34_slope[3:] = (ema34_1w[3:] - ema34_1w[:-3]) / 3  # 3-bar slope
    
    # Align 1w EMA34 and slope to 6h
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    ema34_slope_aligned = align_htf_to_ltf(prices, df_1w, ema34_slope)
    
    # Calculate 1d ATR(14) for volatility regime filter
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR percentile rank (using 50-day lookback for stability)
    atr_percentile = pd.Series(atr_1d).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema34_slope_aligned[i]) or 
            np.isnan(atr_percentile_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # 1d volatility regime: ATR > 50th percentile (avoid low-vol chop)
        vol_regime = atr_percentile_aligned[i] > 50
        
        # Weekly trend filter
        weekly_uptrend = ema34_slope_aligned[i] > 0
        weekly_downtrend = ema34_slope_aligned[i] < 0
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power > 0 (bulls in control) AND Bear Power < 0 (bears weak) 
            #            AND weekly uptrend AND volatility regime
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                weekly_uptrend and vol_regime):
                position = 1
                signals[i] = 0.25
            # Short entry: Bull Power < 0 (bulls weak) AND Bear Power > 0 (bears in control)
            #            AND weekly downtrend AND volatility regime
            elif (bull_power[i] < 0 and bear_power[i] > 0 and 
                  weekly_downtrend and vol_regime):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Elder Ray power crossover (change in bull/bear balance)
            # 2. Volatility collapse (ATR drops below 30th percentile)
            
            if position == 1:  # Long position
                exit_condition = (
                    bull_power[i] <= 0 or  # Bulls lose control
                    atr_percentile_aligned[i] < 30  # Volatility collapse
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    bear_power[i] >= 0 or  # Bears lose control
                    atr_percentile_aligned[i] < 30  # Volatility collapse
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals