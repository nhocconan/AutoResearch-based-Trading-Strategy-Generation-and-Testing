#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume and ATR regime filter
# - Primary: 12h timeframe for lower trade frequency (target: 50-150 trades over 4 years)
# - HTF: 1d for Camarilla pivot levels (H3/L3) and volume confirmation
# - Long: Price breaks above H3 + volume > 1.5x 20-period MA + 1d ATR > 50th percentile
# - Short: Price breaks below L3 + volume > 1.5x 20-period MA + 1d ATR > 50th percentile
# - Exit: Price retouches pivot point (PP) or ATR < 30th percentile (low vol regime)
# - Position sizing: 0.25 (discrete level)
# - Works in bull/bear: Camarilla levels act as support/resistance; ATR/volume filters avoid false signals in ranging markets

name = "12h_1d_camarilla_volume_v3"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
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
    
    # Calculate 1d Camarilla pivot levels
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    # Range = High - Low
    range_1d = high_1d - low_1d
    # Resistance levels: R4 = Close + Range * 1.5/2, R3 = Close + Range * 1.25/2, etc.
    # Standard Camarilla: H3 = Close + Range * 1.1/4, L3 = Close - Range * 1.1/4
    h3_1d = close_1d + (range_1d * 1.1 / 4)
    l3_1d = close_1d - (range_1d * 1.1 / 4)
    
    # Align to 12h timeframe (wait for completed 1d bar)
    pp_12h = align_htf_to_ltf(prices, df_1d, pp_1d)
    h3_12h = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_12h = align_htf_to_ltf(prices, df_1d, l3_1d)
    
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
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(pp_12h[i]) or np.isnan(h3_12h[i]) or np.isnan(l3_12h[i]) or 
            np.isnan(atr_percentile_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # 1d volatility regime: ATR > 50th percentile (avoid low-vol chop)
        vol_regime = atr_percentile_aligned[i] > 50
        
        # Volume confirmation: current 1d volume > 1.5x 20-period MA
        # Get the current 1d volume value (aligned to 12h)
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_spike = volume_1d_aligned[i] > 1.5 * volume_ma_20_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above H3 + vol regime + volume spike
            if (close_12h[i] > h3_12h[i] and vol_regime and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below L3 + vol regime + volume spike
            elif (close_12h[i] < l3_12h[i] and vol_regime and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price retouches pivot point (PP) - mean reversion
            # 2. ATR falls below 30th percentile (low volatility regime)
            
            if position == 1:  # Long position
                exit_condition = (
                    close_12h[i] <= pp_12h[i] or  # Price retraced to or below pivot
                    atr_percentile_aligned[i] < 30  # Low volatility regime
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    close_12h[i] >= pp_12h[i] or  # Price retraced to or above pivot
                    atr_percentile_aligned[i] < 30  # Low volatility regime
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals