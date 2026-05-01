#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted MACD with 1d Regime Filter (Chop/ADX).
# Uses MACD histogram filtered by 1d choppiness index to avoid whipsaws in ranging markets.
# Long when: MACD histogram > 0 AND 1d Chop < 38.2 (trending) AND 1d ADX > 25.
# Short when: MACD histogram < 0 AND 1d Chop < 38.2 AND 1d ADX > 25.
# Uses discrete sizing 0.25. Target: 15-30 trades/year.
# Volume-weighting improves signal quality in low-volume 6h candles.
# Regime filter ensures we only trade when higher timeframe is trending, reducing false signals.

name = "6h_VolMACD_1dChopADX_Regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for regime filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d Choppiness Index: measures if market is ranging (high CHOP) or trending (low CHOP)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr1 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])  # align length
    
    # ATR(14) for 1d
    atr_1d = np.full_like(close_1d, np.nan)
    for i in range(14, len(close_1d)):
        if i == 14:
            atr_1d[i] = np.nanmean(tr1[1:15])  # first ATR is average of first 14 TR
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr1[i]) / 14
    
    # Sum of ATR(14) over 14 periods
    atr_sum_1d = np.full_like(close_1d, np.nan)
    for i in range(27, len(close_1d)):  # 14+13=27 for first valid sum
        atr_sum_1d[i] = np.nansum(atr_1d[i-13:i+1])
    
    # Chop = 100 * log10(sum(ATR14) / (max(high)-min(low)) over period) / log10(period)
    max_high_1d = np.full_like(close_1d, np.nan)
    min_low_1d = np.full_like(close_1d, np.nan)
    for i in range(13, len(close_1d)):
        max_high_1d[i] = np.nanmax(high_1d[i-13:i+1])
        min_low_1d[i] = np.nanmin(low_1d[i-13:i+1])
    
    chop_1d = np.full_like(close_1d, np.nan)
    for i in range(27, len(close_1d)):
        if atr_sum_1d[i] > 0 and (max_high_1d[i] - min_low_1d[i]) > 0:
            chop_1d[i] = 100 * np.log10(atr_sum_1d[i] / (max_high_1d[i] - min_low_1d[i])) / np.log10(14)
    
    # 1d ADX: measures trend strength
    # +DM, -DM, TR
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(low_1d)
    for i in range(1, len(high_1d)):
        plus_dm[i] = max(0, high_1d[i] - high_1d[i-1]) if (high_1d[i] - high_1d[i-1]) > (low_1d[i-1] - low_1d[i]) else 0
        minus_dm[i] = max(0, low_1d[i-1] - low_1d[i]) if (low_1d[i-1] - low_1d[i]) > (high_1d[i] - high_1d[i-1]) else 0
    
    # Smoothed +DM, -DM, TR (using Wilder's smoothing = EMA with alpha=1/period)
    def wilder_smooth(values, period):
        """Wilder's smoothing (similar to EMA with alpha=1/period)"""
        if len(values) < period:
            return np.full_like(values, np.nan)
        smoothed = np.full_like(values, np.nan)
        smoothed[period-1] = np.nanmean(values[:period])
        alpha = 1.0 / period
        for i in range(period, len(values)):
            if not np.isnan(smoothed[i-1]):
                smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i-1]
            else:
                smoothed[i] = np.nan
        return smoothed
    
    tr_1d = tr1  # already calculated
    plus_dm_smooth = wilder_smooth(plus_dm, 14)
    minus_dm_smooth = wilder_smooth(minus_dm, 14)
    tr_smooth = wilder_smooth(tr_1d, 14)
    
    # +DI, -DI
    plus_di_1d = np.full_like(close_1d, np.nan)
    minus_di_1d = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if tr_smooth[i] > 0:
            plus_di_1d[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di_1d[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
    
    # DX and ADX
    dx_1d = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if (plus_di_1d[i] + minus_di_1d[i]) > 0:
            dx_1d[i] = 100 * np.abs(plus_di_1d[i] - minus_di_1d[i]) / (plus_di_1d[i] + minus_di_1d[i])
    
    adx_1d = wilder_smooth(dx_1d, 14)
    
    # Align 1d regime filters to 6h
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume-Weighted MACD on 6h
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    # Volume-weighted price
    vwp = typical_price * volume
    # Cumulative volume and VWP for VWAP-like calculation
    cum_vol = np.nancumsum(volume)
    cum_vwp = np.nancumsum(vwp)
    # Avoid division by zero
    vwap = np.divide(cum_vwp, cum_vol, out=np.full_like(cum_vwp, np.nan), where=cum_vol!=0)
    # Reset VWAP at session start (simplified: reset when volume==0 or new day)
    # For simplicity, we'll use a rolling VWAP approximation
    # Instead, use EWM of typical price weighted by volume
    # VWMA: volume-weighted moving average
    def vwma(values, vol, period):
        """Volume-weighted moving average"""
        if len(values) < period:
            return np.full_like(values, np.nan)
        vwma_vals = np.full_like(values, np.nan)
        for i in range(period-1, len(values)):
            vol_sum = np.nansum(vol[i-period+1:i+1])
            if vol_sum > 0:
                vwap_val = np.nansum(values[i-period+1:i+1] * vol[i-period+1:i+1]) / vol_sum
                vwma_vals[i] = vwap_val
        return vwma_vals
    
    # Fast and slow VWMA for MACD
    vwma_fast = vwma(typical_price, volume, 12)
    vwma_slow = vwma(typical_price, volume, 26)
    macd_line = vwma_fast - vwma_slow
    # Signal line: EWM of MACD
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_histogram = macd_line - signal_line
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
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
        if (np.isnan(macd_histogram[i]) or 
            np.isnan(chop_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_macd_hist = macd_histogram[i]
        curr_chop = chop_1d_aligned[i]
        curr_adx = adx_1d_aligned[i]
        
        # Regime conditions: trending market (low chop, strong trend)
        is_trending = (curr_chop < 38.2) and (curr_adx > 25)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: MACD hist > 0 AND trending regime
            if curr_macd_hist > 0 and is_trending:
                signals[i] = 0.25
                position = 1
            # Short: MACD hist < 0 AND trending regime
            elif curr_macd_hist < 0 and is_trending:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: MACD hist <= 0 OR regime changes to ranging
            if curr_macd_hist <= 0 or not is_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: MACD hist >= 0 OR regime changes to ranging
            if curr_macd_hist >= 0 or not is_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals