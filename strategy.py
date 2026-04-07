#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volatility-Adaptive Breakout with 1d Trend Filter
# Uses ATR-based breakout detection combined with 1-day EMA trend filter.
# In low volatility (range), uses Bollinger Band mean reversion.
# In high volatility (breakout), follows the trend with volume confirmation.
# Designed to work in both bull/bear markets by adapting to volatility regimes.
# Target: 20-40 trades/year to minimize fee impact.

name = "6h_volatility_adaptive_breakout_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ATR for volatility measurement
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Bollinger Bands (20, 2)
    bb_middle = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + (bb_std * 2)
    bb_lower = bb_middle - (bb_std * 2)
    
    # Donchian Channel (20) for breakout
    dc_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    dc_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volatility regime: ATR ratio (current vs 50-period average)
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    vol_ratio = np.where(atr_ma > 0, atr / atr_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or
            np.isnan(vol_ma[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i]
        
        # Trend direction from 1d EMA
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Regime detection: low volatility (range) vs high volatility (trend)
        low_vol = vol_ratio[i] < 0.8
        high_vol = vol_ratio[i] > 1.2
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit on reverse signal or volatility collapse
            if (downtrend and low_vol) or (close[i] < bb_lower[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit on reverse signal or volatility collapse
            if (uptrend and low_vol) or (close[i] > bb_upper[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            if low_vol:
                # Low volatility: mean reversion at Bollinger Bands
                if close[i] <= bb_lower[i] and uptrend and vol_confirm:
                    position = 1
                    signals[i] = 0.25
                elif close[i] >= bb_upper[i] and downtrend and vol_confirm:
                    position = -1
                    signals[i] = -0.25
            elif high_vol:
                # High volatility: breakout in trend direction
                if close[i] > dc_upper[i] and uptrend and vol_confirm:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < dc_lower[i] and downtrend and vol_confirm:
                    position = -1
                    signals[i] = -0.25
    
    return signals