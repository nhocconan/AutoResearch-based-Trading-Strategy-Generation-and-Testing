#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d volume and ATR regime filter
# - Primary: 12h timeframe for lower trade frequency and better trend capture
# - HTF: 1d for volatility (ATR percentile) and volume confirmation
# - Williams Alligator: Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)
# - Long: Lips > Teeth > Jaw (bullish alignment) + 1d ATR > 50th percentile + volume > 1.5x 20-period MA
# - Short: Lips < Teeth < Jaw (bearish alignment) + 1d ATR > 50th percentile + volume > 1.5x 20-period MA
# - Exit: When Alligator lines intertwine (Lips crosses Teeth or Jaw) or ATR < 30th percentile
# - Position sizing: 0.25 (discrete level)
# - Target: 50-150 total trades over 4 years (12-37/year) - within 12h sweet spot
# - Works in bull/bear: Alligator catches strong trends, regime filter avoids chop, volume confirms conviction

name = "12h_1d_alligator_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h OHLCV
    open_12h = prices['open'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    volume_12h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams Alligator SMMA (Smoothed Moving Average)
    def smma(source, period):
        """Smoothed Moving Average - similar to Wilder's smoothing"""
        if len(source) < period:
            return np.full(len(source), np.nan)
        result = np.full(len(source), np.nan)
        # First value is simple SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: (prev * (period-1) + current) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    # Calculate Alligator lines on 12h data
    jaw = smma(close_12h, 13)  # Jaw (Blue) - 13-period SMMA
    teeth = smma(close_12h, 8)  # Teeth (Red) - 8-period SMMA
    lips = smma(close_12h, 5)   # Lips (Green) - 5-period SMMA
    
    # Calculate 1d ATR(14) for volatility regime filter
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR percentile rank (using 30-bar lookback)
    atr_percentile = pd.Series(atr_1d).rolling(window=30, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # 1d volatility regime: ATR > 50th percentile (avoid low-vol chop)
        vol_regime = atr_percentile_aligned[i] > 50
        
        # Volume confirmation: current 1d volume > 1.5x 20-period MA
        volume_spike = volume_1d[i // 1] > 1.5 * volume_ma_20_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bullish Alligator alignment + vol regime + volume spike
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and vol_regime and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Bearish Alligator alignment + vol regime + volume spike
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and vol_regime and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Alligator lines intertwine (Lips crosses Teeth or Jaw) - trend weakening
            # 2. Volatility drops below 30th percentile (choppy market)
            
            lips_cross_teeth = (lips[i] > teeth[i] and lips[i-1] <= teeth[i-1]) or \
                              (lips[i] < teeth[i] and lips[i-1] >= teeth[i-1])
            lips_cross_jaw = (lips[i] > jaw[i] and lips[i-1] <= jaw[i-1]) or \
                            (lips[i] < jaw[i] and lips[i-1] >= jaw[i-1])
            alligator_intertwined = lips_cross_teeth or lips_cross_jaw
            
            low_volatility = atr_percentile_aligned[i] < 30
            
            if alligator_intertwined or low_volatility:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals