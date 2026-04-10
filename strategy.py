#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and ATR regime filter
# - Long when price breaks above Camarilla H3 level (1d) + 1d volume > 1.5x 20-period volume SMA + ATR(14) < ATR(50) (low volatility)
# - Short when price breaks below Camarilla L3 level (1d) + same volume and volatility conditions
# - Exit: price returns to Camarilla Pivot point (1d)
# - Position sizing: 0.25 discrete level
# - Camarilla levels provide high-probability intraday reversal/breakout levels
# - Volume confirmation ensures breakout strength
# - ATR regime filter avoids false breakouts during high volatility
# - Target: 20-40 trades/year to minimize fee drag while capturing strong moves

name = "4h_1d_camarilla_atr_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 1d Camarilla levels (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # H4 = close + 1.5*(high-low), H3 = close + 1.0*(high-low), etc.
    # L3 = close - 1.0*(high-low), L4 = close - 1.5*(high-low)
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + 1.0 * range_1d
    camarilla_l3 = close_1d - 1.0 * range_1d
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3.0  # Standard pivot
    
    # Align Camarilla levels to 4h timeframe (using previous day's values)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Calculate 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 4h ATR for volatility regime filter
    # True Range
    tr1 = np.maximum(high - low, 
                     np.maximum(np.abs(high - np.roll(close, 1)), 
                                np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]
    
    atr_period_short = 14
    atr_period_long = 50
    
    atr_short = pd.Series(tr1).rolling(window=atr_period_short, min_periods=atr_period_short).mean().values
    atr_long = pd.Series(tr1).rolling(window=atr_period_long, min_periods=atr_period_long).mean().values
    
    # ATR ratio: short/long < 1 indicates low volatility regime
    atr_ratio = np.where(atr_long > 0, atr_short / atr_long, 1.0)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(atr_ratio[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume for volume spike confirmation
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        
        # Volume confirmation: current 1d volume > 1.5x 20-period SMA (volume spike)
        vol_confirm = vol_1d_current[i] > 1.5 * volume_sma_20_1d_aligned[i]
        
        # Volatility regime: ATR ratio < 0.95 indicates low volatility (squeeze condition)
        low_vol_regime = atr_ratio[i] < 0.95
        
        # Camarilla breakout signals
        long_entry = (close[i] > camarilla_h3_aligned[i]) and low_vol_regime and vol_confirm
        short_entry = (close[i] < camarilla_l3_aligned[i]) and low_vol_regime and vol_confirm
        exit_long = close[i] < camarilla_pivot_aligned[i]  # Exit long when price crosses below pivot
        exit_short = close[i] > camarilla_pivot_aligned[i]  # Exit short when price crosses above pivot
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals