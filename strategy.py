#!/usr/bin/env python3
# 1h_hma_trend_pullback_v1
# Hypothesis: 1h strategy using 4h HMA(21) for trend direction, 1h RSI(14) pullback to EMA(21) for entry, and volume confirmation (>1.3x 20-bar avg). Uses 1d ADX(14) > 20 to filter ranging markets. Discrete position sizing (0.20) to minimize fee churn. Target: 15-35 trades/year (60-140 total over 4 years). Works in bull/bear: HMA identifies trend, pullback entries capture retracements in strong trends, volume confirms conviction, ADX avoids sideways chop.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_hma_trend_pullback_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h EMA(21) for dynamic support/resistance
    close_s = pd.Series(close)
    ema_21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 1h RSI(14) for pullback detection
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # 1h volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Multi-timeframe: 4h HMA(21) trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # Calculate HMA(21) on 4h close
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    wma_half = pd.Series(close_4h).ewm(span=half_len, adjust=False, min_periods=half_len).mean().values
    wma_full = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_4h = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False, min_periods=sqrt_len).mean().values
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Multi-timeframe: 1d ADX(14) for regime filter (trending >20)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    # Smooth TR, DM+ , DM- with Welles Wilder's smoothing (alpha=1/14)
    tr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_21[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(hma_4h_aligned[i]) or
            np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        # Regime filter: ADX > 20 indicates trending market
        trending_market = adx_aligned[i] > 20
        # HTF trend filter: price relative to 4h HMA(21)
        htf_uptrend = close[i] > hma_4h_aligned[i]
        htf_downtrend = close[i] < hma_4h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: RSI crosses below 40 (momentum loss) or close below EMA(21)
            if rsi_values[i] < 40 or close[i] < ema_21[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI crosses above 60 (momentum loss) or close above EMA(21)
            if rsi_values[i] > 60 or close[i] > ema_21[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Check for pullback entry in trending market
            # Long: RSI < 40 (oversold pullback) in uptrend
            bullish_setup = (rsi_values[i] < 40) and volume_confirmed and trending_market and htf_uptrend
            # Short: RSI > 60 (overbought pullback) in downtrend
            bearish_setup = (rsi_values[i] > 60) and volume_confirmed and trending_market and htf_downtrend
            
            if bullish_setup:
                position = 1
                signals[i] = 0.20
            elif bearish_setup:
                position = -1
                signals[i] = -0.20
    
    return signals