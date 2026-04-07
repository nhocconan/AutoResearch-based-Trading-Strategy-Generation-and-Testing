#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h trend filter and volume confirmation
# Long when price breaks above Donchian(20) high and 12h close > 12h EMA25 (uptrend)
# Short when price breaks below Donchian(20) low and 12h close < 12h EMA25 (downtrend)
# Exit when price crosses opposite Donchian level
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses Donchian channels from 4h price and trend filter from 12h EMA
# Target: 75-200 total trades over 4 years (19-50/year)

name = "4h_donchian20_12h_ema25_vol_v1"
timeframe = "4h"
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
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12-hour EMA(25) for trend filter
    close_12h = df_12h['close'].values
    ema_25_12h = pd.Series(close_12h).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_25_12h)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume moving average (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(ema_25_12h_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below Donchian low (20-period)
            elif close[i] < low[i-20]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above Donchian high (20-period)
            elif close[i] > high[i-20]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Calculate Donchian channels (20-period high/low)
            # Using data up to previous bar to avoid look-ahead
            donchian_high = np.max(high[i-20:i])
            donchian_low = np.min(low[i-20:i])
            
            # Trend filter: 12h EMA25
            uptrend = close[i] > ema_25_12h_aligned[i]
            downtrend = close[i] < ema_25_12h_aligned[i]
            
            # Volume confirmation: current volume > 20-period average
            vol_confirmed = volume[i] > vol_ma[i]
            
            # Long: price breaks above Donchian high in uptrend with volume
            if close[i] > donchian_high and uptrend and vol_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low in downtrend with volume
            elif close[i] < donchian_low and downtrend and vol_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals