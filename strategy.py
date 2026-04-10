#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power + Weekly Trend Filter
# - Primary: 6h timeframe for balance of signal frequency and fee drag
# - HTF: 1w for major trend direction (avoid counter-trend trades)
# - HTF: 1d for volatility regime (ATR filter)
# - Long: Bull Power > 0 AND Bear Power < 0 (bullish momentum) + price > EMA50(1w) + 1d ATR > 30th percentile
# - Short: Bull Power < 0 AND Bear Power > 0 (bearish momentum) + price < EMA50(1w) + 1d ATR > 30th percentile
# - Exit: Opposite Elder Ray signal (Bull Power < 0 for longs, Bear Power > 0 for shorts)
# - Position sizing: 0.25 (discrete level)
# - Target: 80-180 total trades over 4 years (20-45/year) - within 6h sweet spot
# - Works in bull/bear: Elder Ray captures momentum shifts; weekly trend filter avoids major counter-trend moves

name = "6h_1w_1d_elderray_power_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 50 or len(df_1d) < 50:
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
    
    # Calculate 1w EMA50 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Elder Ray Power (6h)
    # Bull Power = High - EMA13(Close)
    # Bear Power = Low - EMA13(Close)
    close_6h_series = pd.Series(close_6h)
    ema13_6h = close_6h_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_6h - ema13_6h
    bear_power = low_6h - ema13_6h
    
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(atr_percentile_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # 1w trend filter: price above/below EMA50
        uptrend = close_6h[i] > ema50_1w_aligned[i]
        downtrend = close_6h[i] < ema50_1w_aligned[i]
        
        # 1d volatility regime: ATR > 30th percentile (avoid low-vol chop)
        vol_regime = atr_percentile_aligned[i] > 30
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power > 0 AND Bear Power < 0 (bullish) + uptrend + vol regime
            if (bull_power[i] > 0 and bear_power[i] < 0 and uptrend and vol_regime):
                position = 1
                signals[i] = 0.25
            # Short entry: Bull Power < 0 AND Bear Power > 0 (bearish) + downtrend + vol regime
            elif (bull_power[i] < 0 and bear_power[i] > 0 and downtrend and vol_regime):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Opposite Elder Ray signal
            if position == 1:  # Long position
                exit_condition = bull_power[i] < 0  # Bull power turned negative
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = bear_power[i] > 0  # Bear power turned positive
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals