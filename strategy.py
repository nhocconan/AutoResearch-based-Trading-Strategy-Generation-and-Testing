#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    """
    Strategy: 6h_SMA_Triple_Ratio_MeanRev
    Hypothesis: Uses ratio of 50SMA to 200SMA (1d) to detect regime, then looks for mean reversion 
    when price deviates from 20EMA (6h) in low volatility (ATR-based) conditions.
    Works in bull (buying dips in uptrend) and bear (selling rallies in downtrend) regimes.
    Targets 20-50 trades/year via strict entry conditions.
    """
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h indicators (primary timeframe) ===
    # EMA20 for mean reversion target
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # ATR(14) for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 1d indicators (HTF for regime and volatility context) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # SMA50 and SMA200 for regime detection (trend strength via ratio)
    close_1d_series = pd.Series(close_1d)
    sma_50_1d = close_1d_series.rolling(window=50, min_periods=50).mean().values
    sma_200_1d = close_1d_series.rolling(window=200, min_periods=200).mean().values
    sma_ratio = sma_50_1d / (sma_200_1d + 1e-10)  # >1 = uptrend bias, <1 = downtrend bias
    
    # ATR(14) 1d for volatility regime filter
    tr1_1d = np.abs(high_1d - low_1d)
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr2_1d[0] = np.inf
    tr3_1d[0] = np.inf
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Align HTF indicators to 6h timeframe
    ema_20_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), ema_20)
    atr_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), atr)
    sma_ratio_aligned = align_htf_to_ltf(prices, df_1d, sma_ratio)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === Session filter: 08-20 UTC (active trading hours) ===
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 200  # Need enough for SMA200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_aligned[i]) or np.isnan(atr_aligned[i]) or 
            np.isnan(sma_ratio_aligned[i]) or np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        ema_20_val = ema_20_aligned[i]
        atr_val = atr_aligned[i]
        sma_ratio_val = sma_ratio_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price returns to EMA20 (mean reversion complete) or volatility too high
            if (price >= ema_20_val) or (atr_val > 1.5 * atr_1d_val):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price returns to EMA20 or volatility too high
            if (price <= ema_20_val) or (atr_val > 1.5 * atr_1d_val):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Only trade during session
            if in_session:
                # Calculate deviation from EMA20 in ATR units
                if atr_val > 0:
                    dev_atr = (price - ema_20_val) / atr_val
                else:
                    dev_atr = 0
                
                # LONG: Price significantly below EMA20 in uptrend regime (SMA50 > SMA200)
                # AND volatility is low (current ATR < 1d ATR) for mean reversion setup
                if (dev_atr < -1.5) and (sma_ratio_val > 1.0) and (atr_val < atr_1d_val):
                    signals[i] = 0.25
                    position = 1
                    continue
                
                # SHORT: Price significantly above EMA20 in downtrend regime (SMA50 < SMA200)
                # AND volatility is low for mean reversion setup
                elif (dev_atr > 1.5) and (sma_ratio_val < 1.0) and (atr_val < atr_1d_val):
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_SMA_Triple_Ratio_MeanRev"
timeframe = "6h"
leverage = 1.0