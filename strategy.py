#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d volume and ATR regime filter
# - Primary: 12h timeframe for lower trade frequency (target: 50-150 trades over 4 years)
# - HTF: 1d for trend (Alligator jaws/teeth/lips) and volume confirmation
# - Long: Price > Alligator lips (green) + jaws > teeth > lips (bullish alignment) + 1d ATR > 50th percentile + volume > 1.5x 20-period MA
# - Short: Price < Alligator lips (green) + jaws < teeth < lips (bearish alignment) + 1d ATR > 50th percentile + volume > 1.5x 20-period MA
# - Exit: Price crosses Alligator teeth (red) or ATR < 30th percentile (low vol regime)
# - Position sizing: 0.25 (discrete level)
# - Works in bull/bear: Alligator identifies trends; ATR/volume filters avoid false signals in ranging markets

name = "12h_1d_alligator_volume_v2"
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
    
    # Calculate 1d Alligator (Williams Alligator: SMAs with specific periods)
    # Jaws: 13-period SMA, shifted 8 bars ahead
    # Teeth: 8-period SMA, shifted 5 bars ahead  
    # Lips: 5-period SMA, shifted 3 bars ahead
    # Note: Using min_periods to ensure proper warmup
    jaws_raw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().shift(8)
    teeth_raw = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().shift(5)
    lips_raw = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().shift(3)
    
    # Align to 12h timeframe (wait for completed 1d bar)
    jaws_12h = align_htf_to_ltf(prices, df_1d, jaws_raw.values)
    teeth_12h = align_htf_to_ltf(prices, df_1d, teeth_raw.values)
    lips_12h = align_htf_to_ltf(prices, df_1d, lips_raw.values)
    
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
        if (np.isnan(jaws_12h[i]) or np.isnan(teeth_12h[i]) or 
            np.isnan(lips_12h[i]) or np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i])):
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
            # Bullish alignment: jaws > teeth > lips
            bullish_align = jaws_12h[i] > teeth_12h[i] > lips_12h[i]
            # Bearish alignment: jaws < teeth < lips
            bearish_align = jaws_12h[i] < teeth_12h[i] < lips_12h[i]
            
            # Long entry: Price > lips + bullish alignment + vol regime + volume spike
            if (close_12h[i] > lips_12h[i] and bullish_align and vol_regime and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Price < lips + bearish alignment + vol regime + volume spike
            elif (close_12h[i] < lips_12h[i] and bearish_align and vol_regime and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price crosses Alligator teeth (red line) - trend weakening
            # 2. ATR falls below 30th percentile (low volatility regime)
            
            if position == 1:  # Long position
                exit_condition = (
                    close_12h[i] < teeth_12h[i] or  # Price crossed below teeth
                    atr_percentile_aligned[i] < 30  # Low volatility regime
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    close_12h[i] > teeth_12h[i] or  # Price crossed above teeth
                    atr_percentile_aligned[i] < 30  # Low volatility regime
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals