#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with 1w volume confirmation and ATR-based stoploss
# - Uses 1d Kaufman Adaptive Moving Average (KAMA) for trend direction
# - Volume confirmation: 1d volume > 1.3x 20-period average to filter weak moves
# - ATR(14) trailing stop at 2.0x ATR from extreme for risk control
# - Position size: 0.25 (25% of capital) - discrete level to minimize fee churn
# - Target: ~7-25 trades/year (30-100 total over 4 years) per 1d strategy guidelines
# - Novelty: KAMA adapts to market noise, reducing whipsaws in ranging markets while catching trends
# - Works in bull markets: catches sustained trends
# - Works in bear markets: adaptive nature reduces false signals during chop, volume confirmation adds filter

name = "1d_1w_kama_volume_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1w indicators
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w Kaufman Adaptive Moving Average (KAMA)
    # Efficiency Ratio (ER) = |Change| / Sum(|Changes|) over period
    # Smoothing Constant (SC) = [ER * (fastest SC - slowest SC) + slowest SC]^2
    # KAMAprev = KAMAprev + SC * (price - KAMAprev)
    close_1w_series = pd.Series(close_1w)
    change = abs(close_1w_series.diff(1))
    volatility = change.rolling(window=10, min_periods=10).sum()
    direction = abs(close_1w_series - close_1w_series.shift(10))
    er = direction / volatility
    er = er.fillna(0)  # Handle division by zero
    fastest_sc = 2 / (2 + 1)  # EMA(2)
    slowest_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fastest_sc - slowest_sc) + slowest_sc) ** 2
    sc = sc.fillna(0)  # Handle NaN
    kama = np.zeros(len(close_1w))
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close_1w[i] - kama[i-1])
    
    # Align KAMA to 1d timeframe (completed 1w bar only)
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # 1w volume > 1.3x 20-period average (volume confirmation)
    volume_1w_series = pd.Series(volume_1w)
    avg_volume_20 = volume_1w_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1w > (1.3 * avg_volume_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1w, volume_spike)
    
    # 1d price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 1d ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or
            np.isnan(atr[i]) or
            atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit condition: price retraces 2.0x ATR from high
            if low[i] <= highest_since_entry - (2.0 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit condition: price retraces 2.0x ATR from low
            if high[i] >= lowest_since_entry + (2.0 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for KAMA breakout with volume confirmation
            # Long: price above KAMA AND volume spike
            if close[i] > kama_aligned[i] and volume_spike_aligned[i]:
                position = 1
                highest_since_entry = high[i]
                lowest_since_entry = high[i]
                signals[i] = 0.25
            # Short: price below KAMA AND volume spike
            elif close[i] < kama_aligned[i] and volume_spike_aligned[i]:
                position = -1
                highest_since_entry = low[i]
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals