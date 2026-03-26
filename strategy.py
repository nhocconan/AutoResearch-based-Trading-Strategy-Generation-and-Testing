#!/usr/bin/env python3
"""
Experiment #022: 6h RSI Extreme + ATR Volatility Regime + 1d KAMA Trend

HYPOTHESIS: RSI extremes (below 30 / above 70) on 6h capture momentum reversals
that align with volatility regime changes. When ATR(7) rises above its EMA,
volatility is expanding — ideal for fading RSI extremes. Combined with 1d KAMA
trend bias, this catches reversals WITH trend, not against it. Works in bull
(rally from oversold) and bear (short rallies to overbought).

TIMEFRAME: 6h primary
HTF: 1d for KAMA trend filter
TARGET: 75-200 total trades over 4 years (19-50/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_rsi_atr_regime_kama_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d KAMA for trend (approximated via EWM for speed)
    kama_period = 10
    delta_1d = np.abs(np.diff(df_1d['close'].values))
    diff_1d = np.insert(delta_1d, 0, np.nan)
    vol_1d = pd.Series(np.abs(diff_1d)).rolling(window=kama_period, min_periods=kama_period).mean().values
    er_1d = np.where(vol_1d > 1e-10, diff_1d / vol_1d, 0.0)
    sc_1d = (er_1d * 0.6667) + 0.1111
    kama_1d_raw = pd.Series(df_1d['close'].values).ewm(span=kama_period, min_periods=kama_period, adjust=False).mean().values
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate 6h indicators
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    atr_ratio = atr_7 / np.where(atr_30 > 1e-10, atr_30, 1e-10)
    
    # Volume confirmation (50-period MA for 6h = ~12.5 days)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # RSI(14) for momentum extremes
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]) or atr_7[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # 1d KAMA trend alignment
        price_above_1d_kama = close[i] > kama_1d_aligned[i]
        
        # ATR volatility regime (expanding = good for reversals)
        atr_regime_expanding = atr_ratio[i] >= 1.0
        
        # Volume confirmation
        vol_confirmed = vol_ratio[i] >= 1.2
        
        # RSI momentum extremes
        rsi_val = rsi[i]
        
        desired_signal = 0.0
        
        # === LONG ENTRY: RSI oversold + expanding ATR + volume + bullish 1d trend ===
        if not in_position:
            if rsi_val < 30 and atr_regime_expanding and vol_confirmed and price_above_1d_kama:
                desired_signal = SIZE
            
            # === SHORT ENTRY: RSI overbought + expanding ATR + volume + bearish 1d trend ===
            if rsi_val > 70 and atr_regime_expanding and vol_confirmed and not price_above_1d_kama:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR) ===
        stoploss_triggered = False