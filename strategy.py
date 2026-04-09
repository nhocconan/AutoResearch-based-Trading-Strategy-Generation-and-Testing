#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot + volume spike + ADX regime filter
# - Primary signal: 4h price touches Camarilla H3/L3 from prior 1d candle
# - Volume confirmation: 4h volume > 1.5x 20-period EMA volume (avoid low-participation touches)
# - Regime filter: 1d ADX > 25 for trending (breakout in ADX direction), ADX < 20 for range (mean reversion at Camarilla levels)
# - In trending markets (ADX > 25): only trade breakouts in direction of 1d ADX trend (using +DI/-DI crossover)
# - In ranging markets (ADX < 20): mean reversion at Camarilla H3/L3 with volume spike
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 20-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - Works in bull/bear: Camarilla provides structure, ADX regime filter adapts to market state

name = "4h_1d_camarilla_vol_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ADX calculation (Wilder's smoothing)
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr[0]  # First value
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.mean(data[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    atr_1d = wilders_smoothing(tr, period)
    plus_di_1d = 100 * wilders_smoothing(plus_dm, period) / atr_1d
    minus_di_1d = 100 * wilders_smoothing(minus_dm, period) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smoothing(dx_1d, period)
    
    # ADX trend direction: +DI > -DI = bullish, -DI > +DI = bearish
    adx_bullish = plus_di_1d > minus_di_1d
    adx_bearish = minus_di_1d > plus_di_1d
    
    # ADX regimes: >25 = trending, <20 = ranging
    adx_trending = adx_1d > 25
    adx_ranging = adx_1d < 20
    
    # 1d Camarilla pivot levels (based on prior day's range)
    camarilla_h3 = close_1d + 1.0 * (high_1d - low_1d)
    camarilla_l3 = close_1d - 1.0 * (high_1d - low_1d)
    camarilla_h4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_l4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align all 1d indicators to 4h timeframe (completed 1d bar only)
    adx_trending_aligned = align_htf_to_ltf(prices, df_1d, adx_trending)
    adx_ranging_aligned = align_htf_to_ltf(prices, df_1d, adx_ranging)
    adx_bullish_aligned = align_htf_to_ltf(prices, df_1d, adx_bullish)
    adx_bearish_aligned = align_htf_to_ltf(prices, df_1d, adx_bearish)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h volume confirmation: volume > 1.5x 20-period EMA volume
    volume_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_trending_aligned[i]) or
            np.isnan(adx_ranging_aligned[i]) or
            np.isnan(adx_bullish_aligned[i]) or
            np.isnan(adx_bearish_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions
            if adx_ranging_aligned[i]:
                # In ranging market: exit at Camarilla L3 (mean reversion target)
                if close[i] <= camarilla_l3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif adx_trending_aligned[i] and adx_bullish_aligned[i]:
                # In bullish trend: exit if trend reverses (DI crossover) or breaks L4
                if (not adx_bullish_aligned[i]) or (close[i] <= camarilla_l4_aligned[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:
                # In bearish trend: exit long immediately (should not be long in bearish trend)
                position = 0
                signals[i] = 0.0
                    
        elif position == -1:  # Short position
            # Exit conditions
            if adx_ranging_aligned[i]:
                # In ranging market: exit at Camarilla H3 (mean reversion target)
                if close[i] >= camarilla_h3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif adx_trending_aligned[i] and adx_bearish_aligned[i]:
                # In bearish trend: exit if trend reverses (DI crossover) or breaks H4
                if (not adx_bearish_aligned[i]) or (close[i] >= camarilla_h4_aligned[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:
                # In bullish trend: exit short immediately (should not be short in bullish trend)
                position = 0
                signals[i] = 0.0
        else:  # Flat
            # Look for Camarilla level touch with volume confirmation
            # Long: price touches or crosses above Camarilla H3 AND volume spike
            if high[i] >= camarilla_h3_aligned[i] and volume_spike[i]:
                # In ranging market: only long if also in ranging regime (mean reversion setup)
                # In trending market: only long if in bullish trend (ADX>25 and +DI>-DI)
                if adx_ranging_aligned[i] or (adx_trending_aligned[i] and adx_bullish_aligned[i]):
                    position = 1
                    signals[i] = 0.25
            # Short: price touches or crosses below Camarilla L3 AND volume spike
            elif low[i] <= camarilla_l3_aligned[i] and volume_spike[i]:
                # In ranging market: only short if also in ranging regime (mean reversion setup)
                # In trending market: only short if in bearish trend (ADX>25 and -DI>+DI)
                if adx_ranging_aligned[i] or (adx_trending_aligned[i] and adx_bearish_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals