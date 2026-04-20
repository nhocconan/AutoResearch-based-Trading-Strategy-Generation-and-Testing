#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_TRIX_Trend_Volume_Control"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d: TRIX (15-period) ===
    close_1d = df_1d['close'].values
    # EMA1
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    # EMA2
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    # EMA3
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    # TRIX = (EMA3 - prev EMA3) / prev EMA3 * 100
    trix_raw = np.diff(ema3, prepend=ema3[0])
    trix = trix_raw / np.where(ema3[:-1] != 0, ema3[:-1], np.nan) * 100
    trix = np.append(trix[0], trix)  # align length
    trix_1d_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # === 1d: 50-period EMA for trend filter ===
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 4h: Price and volume ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # 4-period ATR for stop (calculated on 4h)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    atr = pd.Series(tr).rolling(window=4, min_periods=4).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip outside session
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        # Get values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        trix_val = trix_1d_aligned[i]
        ema50_val = ema50_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        # Skip if any value is NaN
        if np.isnan(trix_val) or np.isnan(ema50_val) or np.isnan(vol_ratio_val) or np.isnan(atr_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        if position == 0:
            # Long: Uptrend + positive TRIX + volume confirmation
            if (close_val > ema50_val and          # Price above 1d EMA50 (uptrend)
                trix_val > 0.0 and                 # Positive TRIX (bullish momentum)
                vol_ratio_val > 1.8):              # Volume confirmation
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: Downtrend + negative TRIX + volume confirmation
            elif (close_val < ema50_val and        # Price below 1d EMA50 (downtrend)
                  trix_val < 0.0 and               # Negative TRIX (bearish momentum)
                  vol_ratio_val > 1.8):            # Volume confirmation
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        
        elif position == 1:
            # Long exit: trend breakdown, momentum fade, or stoploss
            if (close_val < ema50_val or           # Price below 1d EMA50
                trix_val < -0.5 or                 # TRIX turns significantly negative
                vol_ratio_val < 1.0 or             # Low volume (losing momentum)
                close_val <= entry_price - 2.5 * atr_val):  # ATR stoploss
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal, momentum fade, or stoploss
            if (close_val > ema50_val or           # Price above 1d EMA50
                trix_val > 0.5 or                  # TRIX turns significantly positive
                vol_ratio_val < 1.0 or             # Low volume (losing momentum)
                close_val >= entry_price + 2.5 * atr_val):  # ATR stoploss
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals