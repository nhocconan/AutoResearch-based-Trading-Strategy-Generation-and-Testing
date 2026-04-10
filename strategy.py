#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d volume and ATR regime filter
# - Primary: 12h timeframe for lower trade frequency (~12-37/year)
# - HTF: 1d for volume confirmation and ATR volatility regime
# - Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) SMAs
# - Long: Lips > Teeth > Jaw (bullish alignment) + 1d volume > 1.5x 20-period MA + 1d ATR > 50th percentile
# - Short: Lips < Teeth < Jaw (bearish alignment) + same filters
# - Exit: When Alligator lines re-cross (Lips crosses Teeth) or ATR drops below 30th percentile
# - Position sizing: 0.25 (discrete level)
# - Works in bull/bear: Alligator catches trends; volatility regime avoids chop; volume confirms conviction

name = "12h_1d_alligator_volume_v1"
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
    
    # Calculate Williams Alligator on 12h
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars  
    # Lips: 5-period SMMA, shifted 3 bars
    def smma(values, period):
        """Smoothed Moving Average"""
        if len(values) < period:
            return np.full_like(values, np.nan, dtype=float)
        result = np.full_like(values, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    jaw_raw = smma(close_12h, 13)
    teeth_raw = smma(close_12h, 8)
    lips_raw = smma(close_12h, 5)
    
    # Apply shifts (Jaw shifted 8, Teeth shifted 5, Lips shifted 3)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    # First few values will be invalid due to roll, but we'll check for NaN anyway
    
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
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # 1d volatility regime: ATR > 50th percentile (avoid low-vol chop)
        vol_regime = atr_percentile_aligned[i] > 50
        
        # Volume confirmation: current 1d volume > 1.5x 20-period MA
        volume_spike = volume_1d[i // 288] > 1.5 * volume_ma_20_1d_aligned[i]  # 288 = 12h bars per day
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Lips > Teeth > Jaw (bullish alignment) + vol regime + volume spike
            if (lips[i] > teeth[i] > jaw[i] and vol_regime and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Lips < Teeth < Jaw (bearish alignment) + vol regime + volume spike
            elif (lips[i] < teeth[i] < jaw[i] and vol_regime and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Alligator lines re-cross (Lips crosses Teeth) - trend weakening
            # 2. ATR drops below 30th percentile (low volatility environment)
            
            if position == 1:  # Long position
                exit_condition = (
                    lips[i] <= teeth[i] or  # Lips crossed below Teeth
                    atr_percentile_aligned[i] < 30  # Low volatility
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    lips[i] >= teeth[i] or  # Lips crossed above Teeth
                    atr_percentile_aligned[i] < 30  # Low volatility
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals