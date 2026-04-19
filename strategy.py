#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA100 trend filter + volume confirmation + ATR stoploss.
# Donchian breakouts capture momentum in trending markets. Daily EMA100 filters for trend direction.
# Volume confirmation avoids false breakouts. ATR-based stoploss manages risk.
# Works in bull/bear markets: only trades in direction of higher timeframe trend.
# Target: 20-40 trades/year per symbol.
name = "4h_Donchian20_EMA100_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA100 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA100 on daily
    ema_100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Donchian Channel (20) on 4h
    donchian_period = 20
    upper_dc = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_dc = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # ATR(14) for stoploss and volatility filter
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d EMA100 to 4h
    ema_100_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(donchian_period, 100, atr_period, 20)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_dc[i]) or np.isnan(lower_dc[i]) or np.isnan(atr[i]) or 
            np.isnan(ema_100_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = upper_dc[i]
        lower = lower_dc[i]
        ema_100_val = ema_100_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.3 * vol_ma
        
        # Breakout conditions
        bullish_breakout = price > upper
        bearish_breakout = price < lower
        
        if position == 0:
            # Look for Donchian breakout in direction of daily trend
            if bullish_breakout and (price > ema_100_val) and volume_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif bearish_breakout and (price < ema_100_val) and volume_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions: ATR stoploss or mean reversion to Donchian midpoint
            stop_loss = entry_price - 2.0 * atr_val
            donchian_mid = (upper + lower) / 2
            if price < stop_loss or price < donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: ATR stoploss or mean reversion to Donchian midpoint
            stop_loss = entry_price + 2.0 * atr_val
            donchian_mid = (upper + lower) / 2
            if price > stop_loss or price > donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals