#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h volume-weighted reversal with 4h trend filter and 1d volatility regime.
# Uses 1h VWAP for mean reversion entries, 4h EMA50 for trend direction, and 1d ATR ratio for volatility filtering.
# Designed to capture short-term reversals within the dominant trend while avoiding high-volatility chop.
# Target: 15-35 trades/year (60-140 total over 4 years) with strict entry conditions.
name = "1h_VWAP_Reversal_4hEMA50_1dATR_Filter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 4h close
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for ATR-based volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 14-period ATR on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 1h VWAP (typical price * volume)
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.where(vwap_den != 0, vwap_num / vwap_den, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need 60 periods for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or np.isnan(vwap[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema = ema_50_4h_aligned[i]
        atr = atr_14_1d_aligned[i]
        vwap_val = vwap[i]
        
        # Volatility filter: avoid trading when ATR is too high (choppy market)
        atr_ma = np.nanmean(atr_14_1d_aligned[max(0, i-20):i+1]) if i >= 20 else atr
        vol_filter = atr < 1.5 * atr_ma  # Only trade when volatility is below 1.5x recent average
        
        if position == 0:
            # Enter long: price below VWAP AND price > 4h EMA50 (uptrend) AND volatility filter
            if price < vwap_val and price > ema and vol_filter:
                signals[i] = 0.20
                position = 1
            # Enter short: price above VWAP AND price < 4h EMA50 (downtrend) AND volatility filter
            elif price > vwap_val and price < ema and vol_filter:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price crosses above VWAP OR trend reverses (price < 4h EMA50)
            if price > vwap_val or price < ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price crosses below VWAP OR trend reverses (price > 4h EMA50)
            if price < vwap_val or price > ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals