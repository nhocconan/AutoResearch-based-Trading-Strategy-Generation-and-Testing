#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_volume = df_1w['volume'].values
    
    # Calculate weekly Donchian channels (20-period) for structure
    highest_20 = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly ATR(14) for volatility filter
    tr1 = pd.Series(weekly_high - weekly_low)
    tr2 = pd.Series(np.abs(weekly_high - np.concatenate([[weekly_close[0]], weekly_close[:-1]])))
    tr3 = pd.Series(np.abs(weekly_low - np.concatenate([[weekly_close[0]], weekly_close[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to daily timeframe with proper delay
    highest_20_d = align_htf_to_ltf(prices, df_1w, highest_20)
    lowest_20_d = align_htf_to_ltf(prices, df_1w, lowest_20)
    atr_14_d = align_htf_to_ltf(prices, df_1w, atr_14)
    
    # Calculate daily KAMA(10,2,30) for trend direction
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder
    # Proper ER calculation for KAMA
    close_series = pd.Series(close)
    direction = np.abs(close_series.diff(10).fillna(0)).values
    volatility = close_series.diff().abs().rolling(10, min_periods=1).sum().values
    er = np.where(volatility > 0, direction / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate daily RSI(14) for mean reversion
    delta = pd.Series(close).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Calculate daily ATR(14) for stoploss
    tr1_d = pd.Series(high - low)
    tr2_d = pd.Series(np.abs(high - np.concatenate([[close[0]], close[:-1]])))
    tr3_d = pd.Series(np.abs(low - np.concatenate([[close[0]], close[:-1]])))
    tr_d = pd.concat([tr1_d, tr2_d, tr3_d], axis=1).max(axis=1)
    atr_14_d_local = tr_d.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20_d[i]) or np.isnan(lowest_20_d[i]) or np.isnan(atr_14_d[i]) or
            np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(atr_14_d_local[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. Price breaks above weekly Donchian HIGH with KAMA up and RSI not overbought → long
        # 2. Price breaks below weekly Donchian LOW with KAMA down and RSI not oversold → short
        # 3. Volatility filter: ATR > 0.3% of price (avoid low volatility chop)
        # 4. Discrete position sizing: 0.25
        
        # Long conditions: breakout above weekly HIGH
        if (close[i] > highest_20_d[i] and            # Price above weekly Donchian HIGH
            kama[i] > kama[i-1] and                  # KAMA rising (trend up)
            rsi[i] < 70 and                          # Not overbought
            atr_14_d[i] > 0.003 * close[i]):         # Volatility filter
            signals[i] = 0.25
            
        # Short conditions: breakdown below weekly LOW
        elif (close[i] < lowest_20_d[i] and          # Price below weekly Donchian LOW
              kama[i] < kama[i-1] and                # KAMA falling (trend down)
              rsi[i] > 30 and                        # Not oversold
              atr_14_d[i] > 0.003 * close[i]):       # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_WeeklyDonchian_KAMA_RSI_Filter"
timeframe = "1d"
leverage = 1.0