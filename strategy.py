#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 12h EMA trend filter and volume confirmation
# Long when price breaks above upper BB after squeeze (BBW < 20th percentile) + 12h EMA50 uptrend + volume > 1.5x avg
# Short when price breaks below lower BB after squeeze + 12h EMA50 downtrend + volume confirmation
# Uses Bollinger Band width percentile to identify low volatility squeezes that precede breakouts
# Works in both bull/bear markets by capturing expansion phases after contraction
# Target: 12-30 trades/year via strict squeeze + breakout + volume confluence

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
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # === 12h Indicator: EMA50 for trend filter ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === 6h Bollinger Bands (20, 2.0) ===
    bb_period = 20
    bb_std = 2.0
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma_20 + (bb_std * std_20)
    lower_band = sma_20 - (bb_std * std_20)
    bb_width = (upper_band - lower_band) / sma_20  # Normalized BB width
    
    # BB width percentile lookback (50 periods ~ ~6-7 days on 6h)
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=30).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(bb_period, 50) + 20  # BB(20) + BB width percentile(50) + volume(20)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(sma_20[i]) or np.isnan(std_20[i]) or
            np.isnan(bb_width_percentile[i]) or np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === SQUEEZE CONDITION: BB width below 20th percentile (low volatility) ===
        is_squeeze = bb_width_percentile[i] < 0.20
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above upper BB (close > upper_band)
        # 2. Was in squeeze (low volatility breakout)
        # 3. 12h EMA50 uptrend (price > EMA50)
        # 4. Volume confirmation
        if (close[i] > upper_band[i]) and is_squeeze and \
           (close[i] > ema_50_12h_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below lower BB (close < lower_band)
        # 2. Was in squeeze (low volatility breakout)
        # 3. 12h EMA50 downtrend (price < EMA50)
        # 4. Volume confirmation
        elif (close[i] < lower_band[i]) and is_squeeze and \
             (close[i] < ema_50_12h_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_BB_Squeeze_Breakout_12hEMA50_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0