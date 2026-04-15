#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band squeeze breakout with 1d trend filter and volume confirmation
# Long when price breaks above upper BB after low volatility squeeze + 1d EMA50 uptrend + volume spike
# Short when price breaks below lower BB after squeeze + 1d EMA50 downtrend + volume spike
# Uses Bollinger Band width percentile to detect low volatility regimes (squeeze)
# Breakouts from squeeze capture explosive moves in both bull and bear markets
# Volume confirmation (2.0x avg) filters false breakouts, targeting ~25-40 trades/year on 12h

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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicator: EMA50 ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 12h Bollinger Bands (20, 2.0) ===
    bb_period = 20
    bb_std = 2.0
    bb_mid = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_mid + (bb_std_dev * bb_std)
    bb_lower = bb_mid - (bb_std_dev * bb_std)
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band Width percentile (50-period lookback) for squeeze detection
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(bb_period, 50, 20) + 5  # BB(20) + EMA50 + vol(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(bb_width_percentile[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Squeeze condition: BB width in lowest 20th percentile (low volatility)
        is_squeeze = bb_width_percentile[i] <= 20.0
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # === LONG CONDITIONS ===
        # 1. Bollinger Band breakout above upper band (close > upper)
        # 2. Was in squeeze on previous bar (volatility contraction precedes expansion)
        # 3. 1d EMA50 uptrend (close > EMA50)
        # 4. Volume confirmation
        if (close[i] > bb_upper[i]) and \
           (is_squeeze) and \
           (close[i] > ema_50_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Bollinger Band breakout below lower band (close < lower)
        # 2. Was in squeeze on previous bar
        # 3. 1d EMA50 downtrend (close < EMA50)
        # 4. Volume confirmation
        elif (close[i] < bb_lower[i]) and \
             (is_squeeze) and \
             (close[i] < ema_50_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_BB_Squeeze_Breakout_1dEMA50_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0