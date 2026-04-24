#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter, volume spike confirmation, and ATR-based stoploss.
- Primary timeframe: 4h for optimal trade frequency (target: 75-200 trades over 4 years).
- HTF: 1d EMA34 for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Volume: Current 4h volume > 2.0 * 20-period volume MA to confirm breakout strength.
- Entry: Long when price breaks above Donchian(20) high AND 1d EMA34 bullish AND volume spike.
         Short when price breaks below Donchian(20) low AND 1d EMA34 bearish AND volume spike.
- Exit: Opposite Donchian breakout or ATR trailing stop (signal=0 when price moves against position by 2.0*ATR).
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
Donchian breakouts capture momentum, EMA34 filters counter-trend noise, volume confirms legitimacy.
Works in bull markets via breakouts and in bear markets via short breakdowns with trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian(20) channels
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for dynamic stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    df_1d_close = df_1d['close'].values
    ema_34_1d = pd.Series(df_1d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 4h
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current 4h volume > 2.0 * 20-period volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    max_favorable_price = 0.0  # for trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 20, 34)  # Donchian20, ATR14, volume MA20, EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(period20_high[i]) or np.isnan(period20_low[i]) or 
            np.isnan(atr[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        dch_high = period20_high[i]
        dch_low = period20_low[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Check for entry signals with volume spike and trend alignment
            if vol_spike:
                # Bullish: price breaks above Donchian high AND price > 1d EMA34
                if curr_high > dch_high and curr_close > ema_trend:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    max_favorable_price = curr_close
                # Bearish: price breaks below Donchian low AND price < 1d EMA34
                elif curr_low < dch_low and curr_close < ema_trend:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    max_favorable_price = curr_close
        elif position == 1:
            # Long position management
            max_favorable_price = max(max_favorable_price, curr_close)
            # Exit conditions: opposite breakout OR ATR trailing stop
            if curr_low < dch_low or curr_close < (max_favorable_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            max_favorable_price = min(max_favorable_price, curr_close)
            # Exit conditions: opposite breakout OR ATR trailing stop
            if curr_high > dch_high or curr_close > (max_favorable_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA34_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0