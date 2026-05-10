#!/usr/bin/env python3
# 4H_Volatility_Regime_Momentum_Breakout
# Hypothesis: Buy breakouts from low volatility regimes (Bollinger Band squeeze) with momentum confirmation.
# In high volatility regimes (BB width > 80th percentile), use mean reversion at Bollinger Bands.
# Uses 12h trend filter (EMA50) to align with higher timeframe direction.
# Works in bull/bear by adapting to volatility regime and using trend filter for direction.
# Target: 20-40 trades/year per symbol.

name = "4H_Volatility_Regime_Momentum_Breakout"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    volume_s = pd.Series(volume)
    
    # Bollinger Bands (20, 2)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean()
    bb_std = close_s.rolling(window=20, min_periods=20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    bb_percent_b = (close - bb_lower.values) / (bb_upper.values - bb_lower.values + 1e-10)
    
    # Bollinger Band Width percentile (for regime detection)
    bb_width_series = bb_width.values
    bb_width_percentile = pd.Series(bb_width_series).rolling(
        window=50, min_periods=20
    ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False).values
    
    # RSI (14) for momentum
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume confirmation (20-period average)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 12h trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_12h_up = close_12h > ema50_12h
    trend_12h_down = close_12h < ema50_12h
    
    # Align 12h trend to 4h
    trend_12h_up_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_up.astype(float))
    trend_12h_down_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_down.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bb_width_percentile[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or
            np.isnan(trend_12h_up_aligned[i]) or np.isnan(trend_12h_down_aligned[i]) or
            np.isnan(bb_percent_b[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        low_vol_regime = bb_width_percentile[i] < 30  # Bottom 30% = squeeze
        high_vol_regime = bb_width_percentile[i] > 70  # Top 30% = expansion
        rsi_overbought = rsi[i] > 70
        rsi_oversold = rsi[i] < 30
        
        trend_up = trend_12h_up_aligned[i] > 0.5
        trend_down = trend_12h_down_aligned[i] > 0.5
        
        if position == 0:
            # Low volatility regime: breakout in direction of 12h trend with volume
            if low_vol_regime and vol_ratio > 1.5:
                if trend_up and bb_percent_b[i] > 0.98:  # Break above upper BB
                    signals[i] = 0.25
                    position = 1
                elif trend_down and bb_percent_b[i] < 0.02:  # Break below lower BB
                    signals[i] = -0.25
                    position = -1
            # High volatility regime: mean reversion at BB extremes
            elif high_vol_regime:
                if rsi_oversold and bb_percent_b[i] < 0.02:  # Oversold at lower BB
                    signals[i] = 0.20
                    position = 1
                elif rsi_overbought and bb_percent_b[i] > 0.98:  # Overbought at upper BB
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:
            # Exit long: reversal signals or volatility regime change
            if (rsi[i] > 70 and bb_percent_b[i] > 0.95) or \
               (trend_12h_down_aligned[i] > 0.5) or \
               (bb_width_percentile[i] > 80 and bb_percent_b[i] > 0.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: reversal signals or volatility regime change
            if (rsi[i] < 30 and bb_percent_b[i] < 0.05) or \
               (trend_12h_up_aligned[i] > 0.5) or \
               (bb_width_percentile[i] > 80 and bb_percent_b[i] < 0.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals