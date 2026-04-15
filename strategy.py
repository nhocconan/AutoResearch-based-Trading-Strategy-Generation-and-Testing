#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA20 pullback to 4h EMA50 with volume confirmation and session filter
# Long when: price > 1h EMA20 AND price <= 1h EMA20 + 0.5*ATR(1h) (pullback zone) AND 4h EMA50 uptrend AND volume > 1.5x 20-bar avg
# Short when: price < 1h EMA20 AND price >= 1h EMA20 - 0.5*ATR(1h) (pullback zone) AND 4h EMA50 downtrend AND volume > 1.5x 20-bar avg
# Uses 1h timeframe for precise entry timing on pullbacks to 4h trend, reducing whipsaws.
# Session filter (08-20 UTC) avoids low-liquidity periods. Discrete size 0.20 limits drawdown and fee churn.
# Target: 15-30 trades/year/symbol by requiring 4h trend alignment + volume spike + pullback precision.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # === 4h Indicators: EMA50 and ATR(14) ===
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 4h ATR(14) for volatility normalization
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_4h = np.concatenate([[np.nan], tr_4h])  # align length
    atr_14_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # === 1h Indicators: EMA20 and ATR(14) for pullback zone ===
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    tr1h = high[1:] - low[1:]
    tr2h = np.abs(high[1:] - close[:-1])
    tr3h = np.abs(low[1:] - close[:-1])
    tr_1h = np.maximum(tr1h, np.maximum(tr2h, tr3h))
    tr_1h = np.concatenate([[np.nan], tr_1h])
    atr_14_1h = pd.Series(tr_1h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 20, 14) + 5
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(atr_14_4h_aligned[i]) or
            np.isnan(ema_20[i]) or np.isnan(atr_14_1h[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Pullback zone: within 0.5 * ATR(1h) of 1h EMA20
        pullback_upper = ema_20[i] + 0.5 * atr_14_1h[i]
        pullback_lower = ema_20[i] - 0.5 * atr_14_1h[i]
        in_pullback_zone = (close[i] >= pullback_lower) and (close[i] <= pullback_upper)
        
        # === LONG CONDITIONS ===
        # 1. Price above 1h EMA20 (bullish bias)
        # 2. Price in pullback zone to EMA20
        # 3. 4h EMA50 uptrend (price > EMA50)
        # 4. Volume confirmation
        if (close[i] > ema_20[i]) and \
           in_pullback_zone and \
           (close[i] > ema_50_4h_aligned[i]) and \
           vol_confirm:
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. Price below 1h EMA20 (bearish bias)
        # 2. Price in pullback zone to EMA20
        # 3. 4h EMA50 downtrend (price < EMA50)
        # 4. Volume confirmation
        elif (close[i] < ema_20[i]) and \
             in_pullback_zone and \
             (close[i] < ema_50_4h_aligned[i]) and \
             vol_confirm:
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_EMA20_Pullback_4hEMA50_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0