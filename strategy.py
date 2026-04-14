#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Bollinger Band squeeze + RSI mean reversion.
# Bollinger Band width < 20th percentile indicates low volatility (squeeze).
# When squeeze releases, price tends to revert to mean (20-day SMA).
# Long when BB width expands from squeeze AND RSI < 40 (oversold) AND price > SMA20.
# Short when BB width expands from squeeze AND RSI > 60 (overbought) AND price < SMA20.
# Exit when RSI crosses 50 (mean) or BB width contracts back below squeeze threshold.
# Designed to capture mean reversion after low volatility periods in both bull and bear markets.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE for Bollinger Bands and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma_20 = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = sma_20 + bb_std * std_20
    bb_lower = sma_20 - bb_std * std_20
    bb_width = bb_upper - bb_lower
    
    # Calculate BB width percentile (lookback 50 days for squeeze threshold)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else np.nan, raw=False
    ).values
    
    # Squeeze condition: BB width < 20th percentile
    squeeze_threshold = 20
    is_squeeze = bb_width_percentile < squeeze_threshold
    
    # Calculate RSI (14)
    rsi_period = 14
    delta = np.diff(close_1d, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Load 1w data ONCE for trend filter (optional - using 1w close > 50w EMA for bull bias)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Bull/bear regime: 1w close above/below 50-week EMA
    bull_regime = np.zeros_like(close_1w, dtype=bool)
    bear_regime = np.zeros_like(close_1w, dtype=bool)
    bull_regime = close_1w > ema_50
    bear_regime = close_1w < ema_50
    
    # Align indicators to lower timeframe
    sma_20_aligned = align_htf_to_ltf(prices, df_1d, sma_20)
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    is_squeeze_aligned = align_htf_to_ltf(prices, df_1d, is_squeeze.astype(float))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    bull_regime_aligned = align_htf_to_ltf(prices, df_1w, bull_regime.astype(float))
    bear_regime_aligned = align_htf_to_ltf(prices, df_1w, bear_regime.astype(float))
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(bb_period, rsi_period, 50)  # Need BB, RSI, and regime
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(sma_20_aligned[i]) or 
            np.isnan(bb_width_percentile_aligned[i]) or
            np.isnan(rsi_aligned[i]) or
            np.isnan(bull_regime_aligned[i]) or
            np.isnan(bear_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Squeeze release: was in squeeze, now expanding
        was_in_squeeze = i > start and is_squeeze_aligned[i-1]
        is_expanding = bb_width_percentile_aligned[i] > bb_width_percentile_aligned[i-1]
        squeeze_released = was_in_squeeze and is_expanding
        
        if position == 0:
            # Look for mean reversion entries after squeeze release
            # Long: squeeze released, RSI oversold, price above SMA20
            if (squeeze_released and 
                rsi_aligned[i] < 40 and 
                close[i] > sma_20_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: squeeze released, RSI overbought, price below SMA20
            elif (squeeze_released and 
                  rsi_aligned[i] > 60 and 
                  close[i] < sma_20_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI crosses above 50 or BB width contracts back into squeeze
            if (rsi_aligned[i] >= 50 or 
                bb_width_percentile_aligned[i] < squeeze_threshold):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI crosses below 50 or BB width contracts back into squeeze
            if (rsi_aligned[i] <= 50 or 
                bb_width_percentile_aligned[i] < squeeze_threshold):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_BB_Squeeze_RSI_MeanReversion_v1"
timeframe = "1d"
leverage = 1.0