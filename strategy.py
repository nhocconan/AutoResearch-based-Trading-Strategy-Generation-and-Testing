#!/usr/bin/env python3
# 1d_1w_keltner_channel_v2
# Strategy: 1-day Keltner Channel breakout with 1-week ADX trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Keltner Channel breakouts capture volatility expansion. 1-week ADX > 25 confirms strong trend.
# Volume > 1.3x 20-day average confirms institutional participation. Designed for low trade frequency (~15-25/year)
# to minimize fee decay. Works in bull markets via upper band breakouts and bear markets via lower band breakdowns.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_keltner_channel_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-week data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # 1-day Keltner Channel (20-period, ATR multiplier 2)
    atr = pd.Series(high - low).rolling(window=20, min_periods=20).mean().values
    ema_mid = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_band = ema_mid + 2 * atr
    lower_band = ema_mid - 2 * atr
    
    # 1-week ADX (14-period) for trend strength filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range and Directional Movement
    tr1 = pd.Series(high_1w).rolling(2).max() - pd.Series(low_1w).rolling(2).min()
    tr2 = abs(pd.Series(high_1w) - pd.Series(close_1w).shift(1))
    tr3 = abs(pd.Series(low_1w) - pd.Series(close_1w).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    plus_dm = pd.Series(high_1w).diff()
    minus_dm = pd.Series(low_1w).diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smooth TR, +DM, -DM
    tr_smooth = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1/14, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1/14, adjust=False).mean()
    
    # Calculate DI+ and DI-
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # Calculate DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/14, adjust=False).mean()
    adx_1w = adx.values
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # 1-day volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(adx_1w_aligned[i]) or np.isnan(vol_avg_20[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.3x 20-day average
        vol_confirm = volume[i] > 1.3 * vol_avg_20[i]
        
        # Keltner Channel breakout signals
        breakout_up = close[i] > upper_band[i-1]
        breakdown_down = close[i] < lower_band[i-1]
        
        # 1-week ADX trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_1w_aligned[i] > 25
        
        # Entry conditions
        # Long: Close above upper Keltner band AND strong trend AND volume confirmation
        if breakout_up and strong_trend and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Close below lower Keltner band AND strong trend AND volume confirmation
        elif breakdown_down and strong_trend and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite Keltner signal (close below mid for long, above mid for short)
        elif position == 1 and close[i] < ema_mid[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > ema_mid[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals