# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_cci_extreme_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 14 or len(df_1w) < 4:
        return signals
    
    # Calculate 1d CCI (20-period)
    tp_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    sma_tp = pd.Series(tp_1d).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(tp_1d).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    cci_1d = (tp_1d - sma_tp) / (0.015 * mad)
    cci_1d = np.where(mad == 0, 0, cci_1d)
    
    # Calculate 1w CCI (14-period)
    tp_1w = (df_1w['high'].values + df_1w['low'].values + df_1w['close'].values) / 3
    sma_tp_1w = pd.Series(tp_1w).rolling(window=14, min_periods=14).mean().values
    mad_1w = pd.Series(tp_1w).rolling(window=14, min_periods=14).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    cci_1w = (tp_1w - sma_tp_1w) / (0.015 * mad_1w)
    cci_1w = np.where(mad_1w == 0, 0, cci_1w)
    
    # Shift by 1 to use only completed bars (avoid look-ahead)
    cci_1d = np.roll(cci_1d, 1)
    cci_1w = np.roll(cci_1w, 1)
    cci_1d[0] = np.nan
    cci_1w[0] = np.nan
    
    # Align HTF CCI to 6t timeframe
    cci_1d_aligned = align_htf_to_ltf(prices, df_1d, cci_1d)
    cci_1w_aligned = align_htf_to_ltf(prices, df_1w, cci_1w)
    
    # Volume confirmation: volume > 1.3x 20-period average (6t)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Trend filter: 50-period EMA on 6t
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(cci_1d_aligned[i]) or np.isnan(cci_1w_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(ema_50[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.3 * vol_ma
        
        # Trend direction
        uptrend = price_close > ema_50[i]
        downtrend = price_close < ema_50[i]
        
        # CCI extreme levels
        cci_1d_val = cci_1d_aligned[i]
        cci_1w_val = cci_1w_aligned[i]
        
        # Long: CCI < -100 on both 1d and 1w (oversold) + volume + uptrend
        long_signal = volume_confirmed and uptrend and (cci_1d_val < -100) and (cci_1w_val < -100)
        
        # Short: CCI > 100 on both 1d and 1w (overbought) + volume + downtrend
        short_signal = volume_confirmed and downtrend and (cci_1d_val > 100) and (cci_1w_val > 100)
        
        # Exit when CCI returns to neutral zone (-50 to 50) on 1d
        exit_long = position == 1 and (cci_1d_val > -50)
        exit_short = position == -1 and (cci_1d_val < 50)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: CCI extreme readings on both daily and weekly timeframes signal overextended
# market conditions likely to reverse. In bear markets, extreme negative CCI on both
# timeframes often precedes bounces; in bull markets, extreme positive CCI precedes pullbacks.
# Volume confirmation ensures institutional participation. Trend filter (50 EMA) aligns
# with intermediate-term direction to avoid counter-trend traps. The dual timeframe
# requirement (1d AND 1w) increases signal reliability. Exits on return to neutral CCI
# (-50 to 50) capture mean reversion while limiting exposure. Designed for low trade
# frequency (target: 15-35 trades/year) to minimize fee drag in 6h timeframe. Works
# across BTC, ETH, and SOL by identifying exhaustion points in any market regime.