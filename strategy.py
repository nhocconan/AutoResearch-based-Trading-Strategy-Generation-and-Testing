#!/usr/bin/env python3
"""
4h_TRIX_Plus_Volume_Spike_and_Choppiness_Filter_v1
Hypothesis: TRIX (12-period) detects momentum shifts; combined with volume spikes (>2x 20-period average)
and choppiness regime (CMF > 0 or price below VWAP) for mean reversion, this strategy avoids overtrading
while capturing reversals in both bull and bear markets. Uses 4h timeframe with 1d/1w filters for trend context.
Target: 20-40 trades/year by requiring TRIX crossover + volume spike + regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate TRIX (12-period) - smoothed triple EMA
    def calculate_trix(close, period=12):
        ema1 = pd.Series(close).ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        trix = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
        return trix.values
    
    trix = calculate_trix(close, 12)
    
    # Volume spike: >2x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    vol_spike = volume > (2.0 * vol_ma)
    
    # Choppiness filter: use Chaikin Money Flow (20) > 0 OR price below VWAP
    # CMF calculation
    mf_multiplier = np.where((high - low) != 0, ((close - low) - (high - close)) / (high - low), 0)
    mf_volume = mf_multiplier * volume
    cmf = np.full_like(close, np.nan)
    cmf_period = 20
    if len(mf_volume) >= cmf_period:
        for i in range(cmf_period, len(mf_volume)):
            cmf[i] = np.sum(mf_volume[i - cmf_period:i]) / np.sum(volume[i - cmf_period:i])
    
    # VWAP calculation (session-based, reset daily)
    typical_price = (high + low + close) / 3
    vwap = np.full_like(close, np.nan)
    cum_tpv = np.zeros(n)
    cum_vol = np.zeros(n)
    for i in range(n):
        cum_tpv[i] = cum_tpv[i-1] + typical_price[i] * volume[i] if i > 0 else typical_price[i] * volume[i]
        cum_vol[i] = cum_vol[i-1] + volume[i] if i > 0 else volume[i]
        if cum_vol[i] > 0:
            vwap[i] = cum_tpv[i] / cum_vol[i]
    
    price_below_vwap = close < vwap
    
    # Get daily data for trend filter (ADX < 25 for range)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    def calculate_adx(high, low, close, period=14):
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[np.nan], dm_plus])
        dm_minus = np.concatenate([[np.nan], dm_minus])
        
        atr = np.full_like(tr, np.nan)
        dm_plus_smooth = np.full_like(dm_plus, np.nan)
        dm_minus_smooth = np.full_like(dm_minus, np.nan)
        
        if len(tr) >= period:
            atr[period] = np.nanmean(tr[1:period+1])
            dm_plus_smooth[period] = np.nanmean(dm_plus[1:period+1])
            dm_minus_smooth[period] = np.nanmean(dm_minus[1:period+1])
            
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
                dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        di_plus = np.full_like(dm_plus_smooth, np.nan)
        di_minus = np.full_like(dm_minus_smooth, np.nan)
        valid = ~np.isnan(atr) & (atr != 0)
        di_plus[valid] = 100 * dm_plus_smooth[valid] / atr[valid]
        di_minus[valid] = 100 * dm_minus_smooth[valid] / atr[valid]
        
        dx = np.full_like(di_plus, np.nan)
        dx_valid = ~np.isnan(di_plus) & ~np.isnan(di_minus) & ((di_plus + di_minus) != 0)
        dx[dx_valid] = 100 * np.abs(di_plus[dx_valid] - di_minus[dx_valid]) / (di_plus[dx_valid] + di_minus[dx_valid])
        
        adx = np.full_like(dx, np.nan)
        if len(dx) >= period:
            adx[2*period-1] = np.nanmean(dx[period:2*period])
            for i in range(2*period, len(dx)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Get weekly data for trend filter (price above/below EMA200)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    if len(close_1w) >= 200:
        ema_200w = pd.Series(close_1w).ewm(span=200, adjust=False).mean().values
    else:
        ema_200w = np.full_like(close_1w, np.nan)
    
    # Align all higher timeframe data
    adx_1d_4h = align_htf_to_ltf(prices, df_1d, adx_1d)
    ema_200w_4h = align_htf_to_ltf(prices, df_1w, ema_200w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 12, 20) + 1
    
    for i in range(start_idx, n):
        if (np.isnan(trix[i]) or np.isnan(vol_ma[i]) or np.isnan(cmf[i]) or 
            np.isnan(vwap[i]) or np.isnan(adx_1d_4h[i]) or np.isnan(ema_200w_4h[i])):
            signals[i] = 0.0
            continue
        
        # TRIX crossover signals
        trix_cross_up = trix[i] > 0 and trix[i-1] <= 0
        trix_cross_down = trix[i] < 0 and trix[i-1] >= 0
        
        # Regime filter: range/choppy market (ADX < 25) OR contrarian in trend
        range_regime = adx_1d_4h[i] < 25
        # In uptrend: look for mean reversion when price below VWAP
        # In downtrend: look for mean reversion when price above VWAP
        uptrend = close[i] > ema_200w_4h[i]
        mean_reversion_long = uptrend and price_below_vwap[i]
        mean_reversion_short = (~uptrend) and (~price_below_vwap[i])
        
        if position == 0:
            # Long: TRIX bullish cross + volume spike + (range OR mean reversion in uptrend)
            if trix_cross_up and vol_spike[i] and (range_regime or mean_reversion_long):
                signals[i] = 0.25
                position = 1
            # Short: TRIX bearish cross + volume spike + (range OR mean reversion in downtrend)
            elif trix_cross_down and vol_spike[i] and (range_regime or mean_reversion_short):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX turns bearish OR volume dries up OR strong trend emerges
            if trix_cross_down or not vol_spike[i] or adx_1d_4h[i] > 30:
                signals[i] = -0.25  # reverse
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX turns bullish OR volume dries up OR strong trend emerges
            if trix_cross_up or not vol_spike[i] or adx_1d_4h[i] > 30:
                signals[i] = 0.25  # reverse
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_TRIX_Plus_Volume_Spike_and_Choppiness_Filter_v1"
timeframe = "4h"
leverage = 1.0