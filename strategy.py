#!/usr/bin/env python3
"""
Hypothesis: 1h RSI(2) mean reversion with 4h trend filter and volume confirmation.
- Primary timeframe: 1h for execution, HTF: 4h for EMA trend, 1d for volume regime.
- Trend: 4h EMA50 > EMA200 = bullish trend (long bias), EMA50 < EMA200 = bearish trend (short bias).
- Entry: Long when RSI(2) < 10 AND volume > 1.5 * 20-period volume MA (oversold in bull trend).
         Short when RSI(2) > 90 AND volume > 1.5 * 20-period volume MA (overbought in bear trend).
- Exit: RSI(2) > 50 for longs, RSI(2) < 50 for shorts (mean reversion completion).
- Volume filter avoids false signals in low volatility.
- Discrete signal size: 0.20 to limit drawdown and reduce fee churn.
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
- Works in bull: buys dips in uptrend. Works in bear: sells rallies in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter (EMA50/EMA200)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 and EMA200
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    # Trend: 1 = bullish (EMA50 > EMA200), -1 = bearish (EMA50 < EMA200), 0 = unclear
    trend_4h = np.where(ema50_4h > ema200_4h, 1, np.where(ema50_4h < ema200_4h, -1, 0))
    
    # Align 4h trend to 1h
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Get 1d data for volume regime (avoid low volume periods)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d average volume (20-period MA)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    # Volume regime: 1 = high volume (above MA), 0 = low volume
    vol_regime_1d = np.where(volume_1d > vol_ma_1d, 1, 0)
    
    # Align 1d volume regime to 1h
    vol_regime_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_regime_1d)
    
    # Calculate 1h RSI(2) for mean reversion signals
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(span=2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(span=2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1h volume confirmation (volume > 1.5 * 20-period MA)
    volume_ma_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_1h = volume > (1.5 * volume_ma_1h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough bars for 4h EMA200 and 1h volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trend_4h_aligned[i]) or np.isnan(vol_regime_1d_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(volume_spike_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade in high volume regimes (avoid low volume noise)
        if vol_regime_1d_aligned[i] == 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade with volume spike on 1h
        if not volume_spike_1h[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        rsi_val = rsi[i]
        trend = trend_4h_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if trend == 1:  # Bullish trend: look for oversold bounces
                if rsi_val < 10:  # Extremely oversold
                    signals[i] = 0.20
                    position = 1
            elif trend == -1:  # Bearish trend: look for overbought pullbacks
                if rsi_val > 90:  # Extremely overbought
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: RSI returns to neutral (>50) or trend changes
            if rsi_val > 50 or trend != 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI returns to neutral (<50) or trend changes
            if rsi_val < 50 or trend != -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI2_4hEMATrend_1dVolumeRegime_v1"
timeframe = "1h"
leverage = 1.0