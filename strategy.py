# 6h_1d_Pivot_R1S1_Breakout_Volume_ATRFilter
# Hypothesis: 6-hour chart with daily Pivot Point R1/S1 breakout strategy, confirmed by volume spikes and ATR-based volatility filtering.
# Rationale: Daily pivot points represent institutional reference points where price often reacts. Breakouts above R1 (resistance) or below S1 (support) with volume confirmation indicate institutional participation. 
# ATR filter ensures we only trade when volatility is sufficient to sustain breakouts, avoiding false signals in low-volatility environments.
# Works in bull/bear markets because: 1) Pivot points adapt to recent price action, 2) Volume confirmation filters weak breakouts, 3) ATR filter adapts to volatility regimes.
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entry criteria.

name = "6h_1d_Pivot_R1S1_Breakout_Volume_ATRFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for volatility filter
    def calculate_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        atr = np.full_like(tr, np.nan, dtype=np.float64)
        if len(tr) >= period:
            atr[period-1] = np.nanmean(tr[:period])
            for i in range(period, len(tr)):
                if not np.isnan(atr[i-1]):
                    atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr = calculate_atr(high, low, close, 14)
    
    # Daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points: P = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    ph = df_1d['high'].values
    pl = df_1d['low'].values
    pc = df_1d['close'].values
    
    p = (ph + pl + pc) / 3.0
    r1 = 2 * p - pl
    s1 = 2 * p - ph
    
    # Align daily pivot points to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: volume > 2.0 * 20-period average (strict for fewer trades)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 2.0)
    
    # ATR filter: only trade when ATR > 1.5 * 50-period average (avoid low volatility)
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_filter = atr > (atr_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume and ATR confirmation
            if (close[i] > r1_aligned[i] and 
                volume_confirm[i] and 
                atr_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and ATR confirmation
            elif (close[i] < s1_aligned[i] and 
                  volume_confirm[i] and 
                  atr_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below S1 (reversal signal)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above R1 (reversal signal)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals