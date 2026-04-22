#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Daily close for HTF trend and volatility
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1-day EMA34 for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Daily True Range for volatility regime
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 4-hour EMA50 for trend alignment
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 4-hour ATR for entry trigger and stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if any data is not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(ema_50[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema50_val = ema_50[i]
        ema34_1d_val = ema_34_1d_aligned[i]
        atr_val = atr[i]
        atr_1d_val = atr_1d_aligned[i]
        
        # Trend alignment: 4h EMA50 and daily EMA34 must agree
        trend_up = ema50_val > ema34_1d_val
        trend_down = ema50_val < ema34_1d_val
        
        # Volatility filter: only trade when volatility is elevated (trending market)
        # Compare current daily ATR to its 50-period EMA
        atr_ema_1d = pd.Series(atr_1d_aligned).ewm(span=50, adjust=False, min_periods=50).mean().values[i]
        vol_filter = atr_1d_val > atr_ema_1d
        
        if position == 0 and vol_filter:
            # Long: price above EMA50 with bullish trend alignment
            if price > ema50_val and trend_up:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price below EMA50 with bearish trend alignment
            elif price < ema50_val and trend_down:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position != 0:
            # Exit: trend reversal or volatility collapse
            trend_reversal = (position == 1 and ema50_val < ema34_1d_val) or \
                            (position == -1 and ema50_val > ema34_1d_val)
            
            # Volatility collapse: current ATR drops below 50% of previous
            vol_collapse = atr_val < 0.5 * atr[i-1] if i > 0 else False
            
            if trend_reversal or vol_collapse:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_EMA50_DailyEMA34_Trend_Align_Volatility_Filter"
timeframe = "4h"
leverage = 1.0