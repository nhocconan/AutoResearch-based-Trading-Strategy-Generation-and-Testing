#!/usr/bin/env python3
"""
1h Camarilla H3/L3 Breakout + 4h EMA34 Trend + Volume Spike + Session Filter
Hypothesis: Camarilla H3/L3 levels act as strong intraday support/resistance.
Breakouts above H3 or below L3 with 4h EMA34 trend alignment and volume spike capture
institutional participation. Session filter (08-20 UTC) avoids low-liquidity Asian session.
Uses discrete position sizing (0.20) to control drawdown. Target: 15-37 trades/year on 1h.
Uses 4h for signal direction (EMA34 trend), 1h only for entry timing (Camarilla breakout).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA34 trend filter (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:
        return np.zeros(n)
    
    # Calculate 4h EMA34 for trend filter
    ema_34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate daily Camarilla pivots for H3/L3 levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Camarilla: H3 = close + 1.1*(high-low)*1.1/6, L3 = close - 1.1*(high-low)*1.1/6
    # Using formula: H3 = C + (H-L)*1.1/6, L3 = C - (H-L)*1.1/6
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    camarilla_h3 = daily_close + (daily_high - daily_low) * 1.1 / 6
    camarilla_l3 = daily_close - (daily_high - daily_low) * 1.1 / 6
    
    # Align Camarilla levels to 1h (no extra delay needed for pivot points)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate ATR(14) for stoploss
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for data to propagate
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_34 = ema_34_4h_aligned[i]
        h3_level = camarilla_h3_aligned[i]
        l3_level = camarilla_l3_aligned[i]
        atr_val = atr[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        if position == 0:
            # Long: break above H3 AND uptrend AND volume spike
            long_condition = curr_high > h3_level and curr_close > ema_34 and volume_spike
            # Short: break below L3 AND downtrend AND volume spike
            short_condition = curr_low < l3_level and curr_close < ema_34 and volume_spike
            
            if long_condition:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or price falls below EMA34
            if curr_close <= entry_price - 2.0 * atr_val or curr_close < ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or price rises above EMA34
            if curr_close >= entry_price + 2.0 * atr_val or curr_close > ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA34_Trend_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0