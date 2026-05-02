#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band Squeeze Breakout + 1d Volume Regime + ATR Filter
# Targets 75-200 total trades over 4 years (19-50/year) to minimize fee drag
# Bollinger Band squeeze (low volatility) precedes explosive moves in both bull and bear markets
# 1d volume regime filter ensures institutional participation (volume > 1.5x 20-day average)
# ATR-based stoploss and discrete position sizing (0.25) controls risk
# Works in bull via breakout continuation and bear via volatility expansion captures

name = "4h_BBand_Squeeze_Breakout_1dVolRegime_ATR_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for volume MA calculation
        return np.zeros(n)
    
    # Calculate 1d volume regime (volume > 1.5x 20-day average)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean()
    vol_regime_1d = df_1d['volume'].values > (vol_ma_1d.values * 1.5)
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime_1d)
    
    # Calculate 4h Bollinger Bands (20, 2)
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean()
    bb_std = close_s.rolling(window=20, min_periods=20).std()
    bb_upper = bb_middle + (2 * bb_std)
    bb_lower = bb_middle - (2 * bb_std)
    bb_width = (bb_upper - bb_lower) / bb_middle
    
    # Bollinger Band squeeze: width < 20-period average width (low volatility)
    bb_width_ma = bb_width.rolling(window=20, min_periods=20).mean()
    bb_squeeze = bb_width < bb_width_ma
    
    # Calculate 4h ATR(14) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = np.nan
    tr3.iloc[0] = np.nan
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 40  # Need 20 for BB + 20 for BB width MA + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(bb_squeeze.iloc[i]) or np.isnan(vol_regime_aligned[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Only trade during volume regime (institutional participation)
            if vol_regime_aligned[i]:
                # Bollinger Band breakout with squeeze confirmation
                # Long: price breaks above upper BB AND was in squeeze (low volatility)
                # Short: price breaks below lower BB AND was in squeeze (low volatility)
                if (close[i] > bb_upper.iloc[i] and 
                    bb_squeeze.iloc[i]):
                    signals[i] = 0.25
                    position = 1
                elif (close[i] < bb_lower.iloc[i] and 
                      bb_squeeze.iloc[i]):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Dynamic exit: price closes below middle BB OR ATR-based stoploss
            # Stoploss: entry price - 2.5 * ATR (approximated via close-based rule)
            if (close[i] < bb_middle.iloc[i] or 
                close[i] < close[i-1] - 2.5 * atr[i]):  # Simplified close-based stop
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Dynamic exit: price closes above middle BB OR ATR-based stoploss
            # Stoploss: entry price + 2.5 * ATR (approximated via close-based rule)
            if (close[i] > bb_middle.iloc[i] or 
                close[i] > close[i-1] + 2.5 * atr[i]):  # Simplified close-based stop
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals