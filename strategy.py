# 4h_12h_donchian_volume_trend_v1
# Strategy: 4h Donchian breakout with 12h volume and trend confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Uses 4h Donchian breakout (20-period) confirmed by 12h volume expansion and 12h EMA trend.
# Works in bull markets (breakouts above upper band) and bear markets (breakdowns below lower band).
# Low trade frequency target: 20-40 trades/year to minimize fee drag while capturing strong momentum moves.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_donchian_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 4h ATR for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h volume filter: volume > 1.5x 20-period average
    vol_12h = df_12h['volume'].values
    vol_ma_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = vol_12h / vol_ma_20_12h
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    # 12h EMA trend: 50-period EMA
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr_4h[i]) or np.isnan(vol_ratio_12h_aligned[i]) or np.isnan(ema_12h_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        atr = atr_4h[i]
        vol_ratio = vol_ratio_12h_aligned[i]
        ema_trend = ema_12h_aligned[i]
        
        # Volume confirmation: expanded volume
        volume_expanded = vol_ratio > 1.5
        
        # Breakout conditions
        breakout_up = price_close > donchian_high[i]
        breakdown_down = price_close < donchian_low[i]
        
        # Trend filter: price above/below 12h EMA
        uptrend = price_close > ema_trend
        downtrend = price_close < ema_trend
        
        # Long: breakout up + volume expansion + uptrend
        long_signal = breakout_up and volume_expanded and uptrend
        
        # Short: breakdown down + volume expansion + downtrend
        short_signal = breakdown_down and volume_expanded and downtrend
        
        # Stoploss: 2.5 * ATR from entry
        stop_long = position == 1 and price_close < (ema_trend - 2.5 * atr)
        stop_short = position == -1 and price_close > (ema_trend + 2.5 * atr)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and stop_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and stop_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals