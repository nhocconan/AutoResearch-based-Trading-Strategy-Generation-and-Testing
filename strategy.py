#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R extreme + 12h trend filter + volume confirmation
    # Williams %R identifies overbought/oversold conditions; 12h EMA50 provides trend direction
    # Volume confirmation reduces false signals. Works in ranging markets (mean reversion)
    # and trending markets (pullbacks in trend). Target: 12-25 trades/year per symbol.
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for HTF trend and Williams %R
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # 12h EMA50 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 12h Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r_12h = np.where(
        (highest_high_14 - lowest_low_14) != 0,
        ((highest_high_14 - close_12h) / (highest_high_14 - lowest_low_14)) * -100,
        np.nan
    )
    
    # 12h volume average for confirmation
    volume_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h indicators to 6h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    williams_r_12h_aligned = align_htf_to_ltf(prices, df_12h, williams_r_12h)
    volume_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20_12h)
    
    # 6h ATR(14) for dynamic position sizing and stops
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_6h = np.full(n, np.nan)
    for i in range(14, n):
        atr_6h[i] = np.mean(tr[i-14:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(williams_r_12h_aligned[i]) or 
            np.isnan(volume_ma_20_12h_aligned[i]) or np.isnan(atr_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.2 * 20-period average
        vol_confirm = volume[i] > 1.2 * volume_ma_20_12h_aligned[i]
        
        # Williams %R extremes: < -80 oversold, > -20 overbought
        wr = williams_r_12h_aligned[i]
        oversold = wr < -80
        overbought = wr > -20
        
        # Trend filter: price above/below EMA50
        price_above_ema = close[i] > ema50_12h_aligned[i]
        price_below_ema = close[i] < ema50_12h_aligned[i]
        
        # Entry conditions: mean reversion in trend direction
        long_entry = oversold and price_above_ema and vol_confirm
        short_entry = overbought and price_below_ema and vol_confirm
        
        # Exit conditions: opposite extreme or trend change
        long_exit = (wr > -50) or (not price_above_ema)  # Exit at midpoint or trend change
        short_exit = (wr < -50) or (not price_below_ema)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_williams_r_extreme_trend_filter_v1"
timeframe = "6h"
leverage = 1.0