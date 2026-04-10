#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal + 1d institutional trend filter + volume spike
# - Primary signal: Williams %R(14) crosses above -80 (oversold reversal) for longs, below -20 (overbought reversal) for shorts
# - Trend filter: 1d close > EMA(200) for longs, < EMA(200) for shorts (long-term institutional bias)
# - Volume filter: 6h volume > 2.0x 50-period average volume (strong momentum confirmation)
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 2.5x ATR(14) on 6h (wider stop for 6h volatility)
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: Williams %R captures reversals at extremes; trend filter avoids fighting the major trend

name = "6h_1d_williamsr_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 210:
        return np.zeros(n)
    
    # Pre-compute 1d EMA(200) for trend filter
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Pre-compute 6h Williams %R(14)
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    highest_high_14 = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_6h) / (highest_high_14 - lowest_low_14)
    
    # Pre-compute 6h volume spike filter
    volume_6h = prices['volume'].values
    avg_volume_50 = pd.Series(volume_6h).rolling(window=50, min_periods=50).mean().values
    vol_spike = volume_6h > (2.0 * avg_volume_50)
    
    # Pre-compute 6h ATR(14) for stoploss
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_200_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(vol_spike[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R crosses above -20 (overbought) OR stoploss hit
            if williams_r[i] > -20 or close_6h[i] < entry_price - 2.5 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -80 (oversold) OR stoploss hit
            if williams_r[i] < -80 or close_6h[i] > entry_price + 2.5 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R reversal with trend and volume filters
            if vol_spike[i]:
                # Long: Williams %R crosses above -80 from below in uptrend (close > EMA200)
                if williams_r[i] > -80 and williams_r[i-1] <= -80 and close_6h[i] > ema_200_aligned[i]:
                    position = 1
                    entry_price = close_6h[i]
                    signals[i] = 0.25
                # Short: Williams %R crosses below -20 from above in downtrend (close < EMA200)
                elif williams_r[i] < -20 and williams_r[i-1] >= -20 and close_6h[i] < ema_200_aligned[i]:
                    position = -1
                    entry_price = close_6h[i]
                    signals[i] = -0.25
    
    return signals