#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d volume spike filter
# - Uses Williams %R(14) on 12h for extreme readings (< -80 oversold, > -20 overbought)
# - Requires 1d volume > 1.5 * 20-period average for confirmation (avoids chop)
# - Enters long when Williams %R crosses above -80 from below (mean reversion long)
# - Enters short when Williams %R crosses below -20 from above (mean reversion short)
# - Uses ATR(14) for dynamic stoploss (2.0 * ATR) and position sizing (0.25)
# - Works in bull markets via buying dips, in bear via selling rallies
# - Target: 12-25 trades/year on 12h timeframe (50-100 total over 4 years) to avoid fee drag

name = "12h_1d_williamsr_meanrev_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d volume confirmation: volume > 1.5 * 20-period average
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (1.5 * vol_ma_1d)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d)
    
    # Pre-compute 12h Williams %R(14)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    diff = highest_high - lowest_low
    williams_r = np.where(diff != 0, -100 * (highest_high - close) / diff, -50.0)
    
    # Pre-compute 12h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(atr[i]) or atr[i] <= 0 or
            np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: stoploss or mean reversion reversal
            if close[i] < prices['close'][i-1] - 2.0 * atr[i]:  # ATR stop
                position = 0
                signals[i] = 0.0
            elif williams_r[i] > -20:  # Overbought - exit long
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: stoploss or mean reversion reversal
            if close[i] > prices['close'][i-1] + 2.0 * atr[i]:  # ATR stop
                position = 0
                signals[i] = 0.0
            elif williams_r[i] < -80:  # Oversold - exit short
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for mean reversion entries with volume confirmation
            # Long: Williams %R crosses above -80 from below
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                volume_confirm_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: Williams %R crosses below -20 from above
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  volume_confirm_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals