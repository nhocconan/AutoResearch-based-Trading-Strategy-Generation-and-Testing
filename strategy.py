#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power + 1d Regime Filter
# - Primary: 6h timeframe for balance of trade frequency and noise reduction
# - HTF: 1d for trend regime (EMA50/EMA200) and volatility (ATR percentile)
# - Long: 6h Bull Power > 0 AND Bear Power < 0 AND 1d EMA50 > EMA200 (bull trend) AND 1d ATR > 30th percentile
# - Short: 6h Bear Power < 0 AND Bull Power > 0 AND 1d EMA50 < EMA200 (bear trend) AND 1d ATR > 30th percentile
# - Exit: Opposite Elder Ray signal (Bull Power < 0 for long exit, Bear Power > 0 for short exit)
# - Position sizing: 0.25 (discrete level)
# - Target: 80-180 total trades over 4 years (20-45/year) - within 6h sweet spot
# - Works in bull/bear: Elder Ray captures trend strength; regime filter avoids chop

name = "6h_1d_elderray_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 6h OHLC
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 6h Elder Ray Power (Bull/Bear Power)
    # Bull Power = High - EMA13(Close)
    # Bear Power = Low - EMA13(Close)
    close_6h_series = pd.Series(close_6h)
    ema13_6h = close_6h_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_6h - ema13_6h
    bear_power = low_6h - ema13_6h
    
    # Calculate 1d EMA50 and EMA200 for trend regime
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 1d ATR(14) for volatility regime filter
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    close_1d_series = pd.Series(close_1d)
    tr1 = high_1d_series - low_1d_series
    tr2 = abs(high_1d_series - close_1d_series.shift(1))
    tr3 = abs(low_1d_series - close_1d_series.shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR percentile rank (using 30-day lookback)
    atr_percentile = pd.Series(atr_1d).rolling(window=30, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Align HTF indicators to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or
            np.isnan(atr_percentile_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # 1d trend regime: EMA50 > EMA200 for bull, EMA50 < EMA200 for bear
        bull_trend = ema50_1d_aligned[i] > ema200_1d_aligned[i]
        bear_trend = ema50_1d_aligned[i] < ema200_1d_aligned[i]
        
        # 1d volatility regime: ATR > 30th percentile (avoid low-vol chop)
        vol_regime = atr_percentile_aligned[i] > 30
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power > 0 AND Bear Power < 0 AND bull trend AND vol regime
            if (bull_power[i] > 0 and bear_power[i] < 0 and bull_trend and vol_regime):
                position = 1
                signals[i] = 0.25
            # Short entry: Bear Power < 0 AND Bull Power > 0 AND bear trend AND vol regime
            elif (bear_power[i] < 0 and bull_power[i] > 0 and bear_trend and vol_regime):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Opposite Elder Ray signal
            if position == 1:  # Long position
                exit_condition = bull_power[i] < 0  # Bull Power turned negative
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = bear_power[i] > 0  # Bear Power turned positive
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals