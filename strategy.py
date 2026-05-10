# 4h_VWAP_Reversion_Trend_Filter
# Hypothesis: Mean-revert to VWAP when price deviates >2σ in ranging markets (ADX<25), but follow trend when trending (ADX>25). 
# Uses VWAP deviation with Bollinger Bands on deviation, filtered by ADX regime. Works in bull/bear by adapting to market state.
# Designed for 20-40 trades/year to avoid fee drag.

name = "4h_VWAP_Reversion_Trend_Filter"
timeframe = "4h"
leverage = 1.0

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
    
    # Typical price and VWAP calculation
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.cumsum(pv)
    cum_volume = np.cumsum(volume)
    vwap = np.where(cum_volume > 0, cum_pv / cum_volume, typical_price)
    
    # VWAP deviation and its standard deviation (20 periods)
    deviation = typical_price - vwap
    dev_ma = np.full(n, np.nan)
    dev_std = np.full(n, np.nan)
    for i in range(20, n):
        dev_ma[i] = np.nanmean(deviation[i-20:i+1])
        dev_std[i] = np.nanstd(deviation[i-20:i+1])
    
    # Bollinger Bands on deviation (2σ)
    upper_band = dev_ma + 2.0 * dev_std
    lower_band = dev_ma - 2.0 * dev_std
    
    # ADX calculation (14 periods) for regime filter
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.nanmean(tr[i-13:i+1])
    
    # Avoid division by zero
    atr_safe = np.where(atr == 0, np.finfo(float).eps, atr)
    plus_di = 100 * np.convolve(plus_dm, np.ones(14)/14, mode='same') / atr_safe
    minus_di = 100 * np.convolve(minus_dm, np.ones(14)/14, mode='same') / atr_safe
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = np.full(n, np.nan)
    for i in range(27, n):  # 14+13 for smoothing
        adx[i] = np.nanmean(dx[i-13:i+1])
    
    # Get 1d EMA50 for trend filter (higher timeframe)
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(dev_ma[i]) or np.isnan(dev_std[i]) or np.isnan(adx[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Regime-based entry: mean revert in ranging (ADX<25), follow trend in trending (ADX>25)
            if adx[i] < 25:  # Ranging market - mean revert to VWAP
                # Long when price < lower BB of deviation (undervalued)
                if deviation[i] < lower_band[i]:
                    # Additional filter: price above 1d EMA50 for quality long
                    if close[i] > ema_50_1d_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                # Short when price > upper BB of deviation (overvalued)
                elif deviation[i] > upper_band[i]:
                    # Additional filter: price below 1d EMA50 for quality short
                    if close[i] < ema_50_1d_aligned[i]:
                        signals[i] = -0.25
                        position = -1
            else:  # Trending market - follow trend
                # Long in uptrend (price > 1d EMA50) on pullback to VWAP
                if close[i] > ema_50_1d_aligned[i] and deviation[i] < 0:
                    if deviation[i] > lower_band[i]:  # Not too extreme, waiting for mean reversion
                        signals[i] = 0.20
                        position = 1
                # Short in downtrend (price < 1d EMA50) on pullback to VWAP
                elif close[i] < ema_50_1d_aligned[i] and deviation[i] > 0:
                    if deviation[i] < upper_band[i]:  # Not too extreme, waiting for mean reversion
                        signals[i] = -0.20
                        position = -1
        
        elif position == 1:
            # Exit: price returns to VWAP or stoploss
            if deviation[i] * deviation[i-1] <= 0:  # Crossed VWAP
                signals[i] = 0.0
                position = 0
            elif close[i] < vwap[i] - 1.5 * (vwap[i] - lower_band[i]):  # Stoploss: 1.5x deviation range
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to VWAP or stoploss
            if deviation[i] * deviation[i-1] <= 0:  # Crossed VWAP
                signals[i] = 0.0
                position = 0
            elif close[i] > vwap[i] + 1.5 * (upper_band[i] - vwap[i]):  # Stoploss: 1.5x deviation range
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals