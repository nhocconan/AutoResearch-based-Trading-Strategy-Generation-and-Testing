#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour volume-weighted breakout with 4-hour trend filter and 1-day volume confirmation
# Long when: price breaks above 1h Donchian(20) high, 4h EMA(20) uptrend, volume > 2x 1h volume average
# Short when: price breaks below 1h Donchian(20) low, 4h EMA(20) downtrend, volume > 2x 1h volume average
# Exit on opposite Donchian break or 4h EMA trend reversal
# Stoploss at 2 * ATR(14)
# Position size: 0.20 (20% of capital)
# Uses 4h for trend direction, 1d for volume regime filter, 1h for precise entry timing
# Target: 60-150 trades over 4 years (15-37/year) to avoid fee drag

name = "1h_donchian20_4h_ema_1d_vol_filter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # 4h EMA(20) for trend filter
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d data for volume regime filter (high/low volume environment)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    volume_1d_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_1d_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_ma)
    
    # 1h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # 1h volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC (avoid low-volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available or outside session
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Donchian low or 4h EMA turns down
            elif close[i] < donchian_low[i] or close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 2 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Donchian high or 4h EMA turns up
            elif close[i] > donchian_high[i] or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with volume confirmation and trend alignment
            # Volume regime: only trade when 1d volume is above average (high volatility environment)
            vol_regime = volume_1d_ma_aligned[i] > 0 and volume[i] > volume_1d_ma_aligned[i]
            
            # Long: price breaks above Donchian high, 4h EMA uptrend, volume spike, in session, high vol regime
            if (close[i] > donchian_high[i] and
                close[i] > ema_4h_aligned[i] and
                volume[i] > 2.0 * volume_ma[i] and
                vol_regime):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low, 4h EMA downtrend, volume spike, in session, high vol regime
            elif (close[i] < donchian_low[i] and
                  close[i] < ema_4h_aligned[i] and
                  volume[i] > 2.0 * volume_ma[i] and
                  vol_regime):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals