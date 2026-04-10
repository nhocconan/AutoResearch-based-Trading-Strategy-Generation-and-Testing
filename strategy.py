#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d volume confirmation and ATR volatility filter
# - Primary: 4h Williams %R(14) for overbought/oversold conditions
# - Volume filter: 1d volume > 1.5x 20-period volume MA to ensure participation
# - Volatility filter: ATR(14) < 0.04 * close to avoid extreme volatility periods
# - Exit: Williams %R returns to -50 level (mean reversion)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Williams %R identifies reversals, volume confirms strength, volatility filter avoids noise
# - Target: 75-200 total trades over 4 years = 19-50/year for 4h timeframe

name = "4h_1d_williamsr_volume_volatility_v1"
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
    
    # Pre-compute HTF data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 14-period Williams %R
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    # Handle division by zero
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Calculate 1d volume confirmation: volume > 1.5x 20-period volume MA
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate 14-period ATR for volatility filter
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    
    # Handle first element
    high_low[0] = high[0] - low[0]
    high_close[0] = np.abs(high[0] - close[0])
    low_close[0] = np.abs(low[0] - close[0])
    
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    volatility_filter = atr < 0.04 * close  # ATR less than 4% of price
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high_14[i]) or np.isnan(lowest_low_14[i]) or 
            np.isnan(williams_r[i]) or np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 1.5x 20-period volume MA
        volume_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_confirm = volume_1d_current[i] > 1.5 * volume_ma_20_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R oversold (< -80) + vol confirmation + volatility filter
            if (williams_r[i] < -80 and 
                vol_confirm and volatility_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Williams %R overbought (> -20) + vol confirmation + volatility filter
            elif (williams_r[i] > -20 and 
                  vol_confirm and volatility_filter[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to mean reversion
            # Exit: Williams %R returns to -50 level (mean reversion)
            if position == 1:  # Long position
                if williams_r[i] >= -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if williams_r[i] <= -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals