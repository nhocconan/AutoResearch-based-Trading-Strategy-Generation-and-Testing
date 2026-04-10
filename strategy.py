#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R Mean Reversion with 1d volume and ATR regime filter
# - Primary: 4h timeframe for optimal trade frequency (19-50/year target)
# - HTF: 1d for volume confirmation and volatility regime (ATR percentile)
# - Long: Williams %R < -80 (oversold) + 1d ATR > 40th percentile + volume > 1.3x 20-period MA
# - Short: Williams %R > -20 (overbought) + 1d ATR > 40th percentile + volume > 1.3x 20-period MA
# - Exit: Williams %R crosses above -50 (for longs) or below -50 (for shorts)
# - Position sizing: 0.25 (discrete level)
# - Target: 75-200 total trades over 4 years (19-50/year) - within 4h sweet spot
# - Works in bull/bear: Williams %R captures mean reversion in ranging markets (2025) and extremes in trending markets

name = "4h_1d_williamsr_mean_reversion_v1"
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
    
    # Pre-compute 4h OHLC
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
    
    # Calculate 4h Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_4h = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low_4h = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r_4h = (highest_high_4h - close_4h) / (highest_high_4h - lowest_low_4h) * -100
    # Handle division by zero (when high == low)
    williams_r_4h = np.where((highest_high_4h - lowest_low_4h) == 0, -50, williams_r_4h)
    
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
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(williams_r_4h[i]) or 
            np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # 1d volatility regime: ATR > 40th percentile (avoid low-vol chop)
        vol_regime = atr_percentile_aligned[i] > 40
        
        # Volume confirmation: current 1d volume > 1.3x 20-period MA
        volume_spike = volume_1d[i] > 1.3 * volume_ma_20_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R < -80 (oversold) + vol regime + volume spike
            if (williams_r_4h[i] < -80 and vol_regime and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Williams %R > -20 (overbought) + vol regime + volume spike
            elif (williams_r_4h[i] > -20 and vol_regime and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Williams %R crosses above -50 (for longs) or below -50 (for shorts)
            if position == 1:  # Long position
                exit_condition = williams_r_4h[i] > -50  # Cross above -50
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = williams_r_4h[i] < -50  # Cross below -50
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals