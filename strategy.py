#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Width Regime + RSI Divergence with 12h EMA Trend Filter
# Uses BBW percentile to identify range (high BBW) vs trend (low BBW) regimes.
# In range (BBW > 60th percentile): mean reversion at RSI extremes (30/70).
# In trend (BBW < 40th percentile): trend following with RSI pullbacks to EMA.
# 12h EMA provides higher timeframe trend filter to avoid counter-trend trades.
# Works in bull/bear by adapting to volatility regime - avoids whipsaws in high volatility,
# captures mean reversion in ranging markets, and trends with the higher timeframe.
# Target: 15-25 trades/year with discrete sizing 0.25.

name = "6h_BBWRegime_RSIDivergence_12hEMAFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA(34) for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Bollinger Bands (20, 2) on 6h for regime detection
    bb_period = 20
    bb_std = 2.0
    
    # Calculate BB middle (SMA), upper, lower
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_20 + (bb_std_dev * bb_std)
    lower_bb = sma_20 - (bb_std_dev * bb_std)
    bb_width = upper_bb - lower_bb
    
    # BB Width percentile (using 50-period lookback) for regime classification
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else np.nan, raw=False
    ).values
    
    # RSI(14) for mean reversion signals
    rsi_period = 14
    delta = pd.Series(close).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period, min_periods=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period, min_periods=rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for BB width percentile and RSI
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(sma_20[i]) or np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or
            np.isnan(bb_width_percentile[i]) or np.isnan(rsi_values[i]) or
            np.isnan(ema_34_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_sma = sma_20[i]
        curr_upper = upper_bb[i]
        curr_lower = lower_bb[i]
        curr_bbw_percentile = bb_width_percentile[i]
        curr_rsi = rsi_values[i]
        curr_ema_12h = ema_34_12h_aligned[i]
        
        # Regime classification based on BB Width percentile
        # High BBW (>60) = ranging market (mean reversion)
        # Low BBW (<40) = trending market (trend following)
        # Middle (40-60) = transition (no trades)
        is_ranging = curr_bbw_percentile > 60
        is_trending = curr_bbw_percentile < 40
        
        if position == 0:  # Flat - look for new entries
            if is_ranging:
                # Mean reversion in ranging market
                # Long: RSI < 30 (oversold) and price near lower BB
                if curr_rsi < 30 and curr_close < curr_sma:
                    signals[i] = 0.25
                    position = 1
                # Short: RSI > 70 (overbought) and price near upper BB
                elif curr_rsi > 70 and curr_close > curr_sma:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif is_trending:
                # Trend following in trending market
                # Long: price > 12h EMA AND RSI pulling back from oversold (30->40)
                if curr_close > curr_ema_12h and 30 <= curr_rsi <= 40:
                    signals[i] = 0.25
                    position = 1
                # Short: price < 12h EMA AND RSI pulling back from overbought (60->70)
                elif curr_close < curr_ema_12h and 60 <= curr_rsi <= 70:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Transition regime - no trades
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions
            if is_ranging:
                # Exit mean reversion: RSI > 50 or price hits upper BB
                if curr_rsi > 50 or curr_close >= curr_upper_bb:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif is_trending:
                # Exit trend follow: RSI > 60 (overbought) or price < 12h EMA
                if curr_rsi > 60 or curr_close < curr_ema_12h:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # Exit in transition
                signals[i] = 0.0
                position = 0
        
        elif position == -1:  # Short position
            # Exit conditions
            if is_ranging:
                # Exit mean reversion: RSI < 50 or price hits lower BB
                if curr_rsi < 50 or curr_close <= curr_lower_bb:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            elif is_trending:
                # Exit trend follow: RSI < 40 (oversold) or price > 12h EMA
                if curr_rsi < 40 or curr_close > curr_ema_12h:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # Exit in transition
                signals[i] = 0.0
                position = 0
    
    return signals