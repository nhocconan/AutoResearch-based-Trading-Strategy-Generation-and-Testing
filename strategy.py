#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d volatility regime filter
# Williams %R identifies overbought/oversold conditions with high sensitivity to momentum
# ATR-based regime filter adapts to volatility: high vol = mean reversion, low vol = trend following
# Works in bull/bear as regime filter adjusts to market conditions
# Target: 15-25 trades/year per symbol (60-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for volatility regime
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ATR for volatility regime (14-period)
    atr_len = 14
    tr1 = pd.Series(high).rolling(window=2).max() - pd.Series(low).rolling(window=2).min()
    tr2 = abs(pd.Series(high) - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low) - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=atr_len, min_periods=atr_len).mean().values
    
    # Calculate ATR volatility regime: current ATR vs 50-period average
    atr_ma_50 = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_1d / atr_ma_50  # >1 = high volatility, <1 = low volatility
    
    # Align ATR ratio to 6h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Williams %R (14-period) on 6h
    wr_len = 14
    highest_high = pd.Series(high).rolling(window=wr_len, min_periods=wr_len).max()
    lowest_low = pd.Series(low).rolling(window=wr_len, min_periods=wr_len).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    wr_values = wr.values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(100, wr_len, atr_len, 50)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(wr_values[i]) or 
            np.isnan(atr_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        wr_val = wr_values[i]
        vol_regime = atr_ratio_aligned[i]
        
        if position == 0:
            # Mean reversion in high volatility: sell at overbought, buy at oversold
            if vol_regime > 1.2:  # High volatility regime
                if wr_val > -20:  # Overbought
                    position = -1
                    signals[i] = -position_size
                elif wr_val < -80:  # Oversold
                    position = 1
                    signals[i] = position_size
                else:
                    signals[i] = 0.0
            # Trend following in low volatility: buy strength, sell weakness
            elif vol_regime < 0.8:  # Low volatility regime
                if wr_val > -50:  # Rising momentum
                    position = 1
                    signals[i] = position_size
                elif wr_val < -50:  # Falling momentum
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to neutral or overbought in high vol
            if vol_regime > 1.2 and wr_val > -50:  # High vol mean reversion exit
                position = 0
                signals[i] = 0.0
            elif vol_regime < 0.8 and wr_val < -80:  # Low vol trend exhaustion
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns to neutral or oversold in high vol
            if vol_regime > 1.2 and wr_val < -50:  # High vol mean reversion exit
                position = 0
                signals[i] = 0.0
            elif vol_regime < 0.8 and wr_val > -20:  # Low vol trend exhaustion
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_WilliamsR_VolatilityRegime_v1"
timeframe = "6h"
leverage = 1.0