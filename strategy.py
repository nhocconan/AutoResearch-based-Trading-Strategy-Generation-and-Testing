#!/usr/bin/env python3

"""
Hypothesis: 4-hour Volume-Weighted Average Price (VWAP) mean reversion with daily volatility regime filter.
Long when price deviates >1.5 standard deviations below VWAP during low volatility (VIX-like regime),
short when price deviates >1.5 standard deviations above VWAP during low volatility.
Exit when price returns to VWAP or volatility expands. Designed for 4h timeframe to capture
mean reversion in ranging markets while avoiding trending periods. Works in both bull and bear
markets by fading extremes during low volatility regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Typical price for VWAP calculation
    typical_price = (high + low + close) / 3.0
    
    # VWAP (20-period)
    pv = typical_price * volume
    cum_pv = np.nancumsum(pv)
    cum_vol = np.nancumsum(volume)
    vwap = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    
    # Standard deviation of price deviation from VWAP (20-period)
    price_dev = typical_price - vwap
    price_dev_series = pd.Series(price_dev)
    dev_std = price_dev_series.rolling(window=20, min_periods=20).std().values
    
    # Daily volatility regime: ATR(14) percentile (50-period lookback)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_pct = pd.Series(atr).rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    # Load daily data for trend filter - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 10:
        return np.zeros(n)
    
    # Daily EMA50 for trend direction (avoid trading against strong trend)
    daily_close = df_daily['close'].values
    ema50_daily = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(vwap[i]) or np.isnan(dev_std[i]) or 
            np.isnan(atr_pct[i]) or np.isnan(ema50_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Low volatility regime: ATR percentile < 40% (avoid high volatility trending periods)
        low_vol = atr_pct[i] < 0.4
        
        # VWAP deviation in standard deviations
        if dev_std[i] > 0:
            dev_sigma = price_dev[i] / dev_std[i]
        else:
            dev_sigma = 0
        
        if position == 0:
            # Long: low vol + price < -1.5 sigma below VWAP + not in strong downtrend
            if low_vol and dev_sigma < -1.5 and ema50_daily_aligned[i] > ema50_daily_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: low vol + price > +1.5 sigma above VWAP + not in strong uptrend
            elif low_vol and dev_sigma > 1.5 and ema50_daily_aligned[i] < ema50_daily_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to VWAP (±0.5 sigma) or volatility expands
            exit_signal = False
            
            if position == 1:
                # Exit long: price >= VWAP - 0.5 sigma or high volatility
                if dev_sigma >= -0.5 or atr_pct[i] > 0.6:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price <= VWAP + 0.5 sigma or high volatility
                if dev_sigma <= 0.5 or atr_pct[i] > 0.6:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_VWAP_MeanReversion_VolRegime_DailyTrend"
timeframe = "4h"
leverage = 1.0