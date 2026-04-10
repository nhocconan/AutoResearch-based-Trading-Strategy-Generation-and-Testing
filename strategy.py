#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted MACD with 12h regime filter and 1d trend confirmation
# - Volume-weighted MACD: MACD line weighted by volume to filter weak momentum
# - Regime filter: 12h ADX > 20 for trending markets, < 15 for ranging
# - Trend confirmation: price > 1d EMA50 for longs, < 1d EMA50 for shorts
# - Entry logic: 
#   * Trending market (ADX > 20): Long when VW-MACD crosses above signal AND price > EMA50
#                                 Short when VW-MACD crosses below signal AND price < EMA50
#   * Ranging market (ADX < 15): Long when VW-MACD < -0.5 AND price near 1d VWAP
#                                Short when VW-MACD > 0.5 AND price near 1d VWAP
# - ATR(14) trailing stop (2.5x) on 6h timeframe
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 20-30 trades/year (80-120 total over 4 years) to stay within HARD MAX: 300 total
# - Volume-weighting reduces false signals from low-volume moves

name = "6h_12h_vwmacd_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h ADX for regime filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr_12h = np.maximum.reduce([tr1, tr2, tr3])
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), 
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)), 
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+
    tr_period = 14
    tr_smooth = pd.Series(tr_12h).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_12h = pd.Series(dx).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    adx_12h = np.where((di_plus + di_minus) == 0, 0, adx_12h)  # avoid division by zero
    
    # Pre-compute 1d EMA50 for trend confirmation
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Pre-compute 1d VWAP for ranging market reference
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    volume_1d = df_1d['volume'].values
    vwap_num = pd.Series(typical_price_1d * volume_1d).expanding().sum().values
    vwap_den = pd.Series(volume_1d).expanding().sum().values
    vwap_1d = np.where(vwap_den != 0, vwap_num / vwap_den, typical_price_1d)
    
    # Align HTF indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Pre-compute 6h Volume-Weighted MACD
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Volume-weighted EMA calculation
    def vwema(values, volumes, span):
        """Volume-weighted exponential moving average"""
        weights = volumes
        weighted_values = values * weights
        ema_weights = pd.Series(weights).ewm(span=span, adjust=False).mean().values
        ema_values = pd.Series(weighted_values).ewm(span=span, adjust=False).mean().values
        # Avoid division by zero
        return np.where(ema_weights != 0, ema_values / ema_weights, values)
    
    # Calculate VW-EMA12 and VW-EMA26
    vw_ema_12 = vwema(close_6h, volume_6h, 12)
    vw_ema_26 = vwema(close_6h, volume_6h, 26)
    
    # MACD line = VW-EMA12 - VW-EMA26
    macd_line = vw_ema_12 - vw_ema_26
    
    # Signal line = VW-EMA9 of MACD line
    signal_line = vwema(macd_line, volume_6h, 9)
    
    # MACD histogram
    macd_hist = macd_line - signal_line
    
    # Align VW-MACD components (already on 6h, but ensure alignment for consistency)
    macd_line_aligned = macd_line  # already 6h
    signal_line_aligned = signal_line  # already 6h
    macd_hist_aligned = macd_hist  # already 6h
    
    # Pre-compute 6h ATR for trailing stop
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    tr1_6h = high_6h - low_6h
    tr2_6h = np.abs(high_6h - np.roll(close_6h, 1))
    tr3_6h = np.abs(low_6h - np.roll(close_6h, 1))
    tr1_6h[0] = np.nan
    tr2_6h[0] = np.nan
    tr3_6h[0] = np.nan
    tr_6h = np.maximum.reduce([tr1_6h, tr2_6h, tr3_6h])
    atr_6h = pd.Series(tr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0  # for trailing stop
    lowest_since_entry = 0.0   # for trailing stop
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vwap_aligned[i]) or np.isnan(macd_line_aligned[i]) or 
            np.isnan(signal_line_aligned[i]) or np.isnan(atr_6h[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Regime filter
        strong_trend = adx_aligned[i] > 20
        ranging_market = adx_aligned[i] < 15
        
        # Trend confirmation
        price_above_ema = close_6h[i] > ema_50_aligned[i]
        price_below_ema = close_6h[i] < ema_50_aligned[i]
        
        # VW-MACD conditions
        macd_above_signal = macd_line_aligned[i] > signal_line_aligned[i]
        macd_below_signal = macd_line_aligned[i] < signal_line_aligned[i]
        macd_hist_positive = macd_hist_aligned[i] > 0
        macd_hist_negative = macd_hist_aligned[i] < 0
        
        # Price relative to VWAP for ranging market
        price_near_vwap = abs(close_6h[i] - vwap_aligned[i]) < (atr_6h[i] * 1.5)
        
        close_price = close_6h[i]
        
        if position == 0:  # Flat - look for new entries
            # Trending market entries (ADX > 20)
            if strong_trend:
                # Long: VW-MACD bullish crossover AND price > EMA50
                if macd_above_signal and not macd_above_signal and i > 0 and macd_line_aligned[i-1] <= signal_line_aligned[i-1] and price_above_ema:
                    position = 1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    highest_since_entry = prices['high'].iloc[i]
                    signals[i] = 0.25
                # Short: VW-MACD bearish crossover AND price < EMA50
                elif macd_below_signal and not macd_below_signal and i > 0 and macd_line_aligned[i-1] >= signal_line_aligned[i-1] and price_below_ema:
                    position = -1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    lowest_since_entry = prices['low'].iloc[i]
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
            # Ranging market mean reversion (ADX < 15)
            elif ranging_market:
                # Long: VW-MACD deeply negative AND price near VWAP
                if macd_line_aligned[i] < -0.5 and price_near_vwap:
                    position = 1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    highest_since_entry = prices['high'].iloc[i]
                    signals[i] = 0.25
                # Short: VW-MACD deeply positive AND price near VWAP
                elif macd_line_aligned[i] > 0.5 and price_near_vwap:
                    position = -1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    lowest_since_entry = prices['low'].iloc[i]
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or trailing stop
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, prices['high'].iloc[i])
                # ATR trailing stop: exit when price drops 2.5*ATR from highest point
                trailing_stop = prices['close'].iloc[i] < highest_since_entry - 2.5 * atr_6h[i]
                exit_condition = trailing_stop
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                # ATR trailing stop: exit when price rises 2.5*ATR from lowest point
                trailing_stop = prices['close'].iloc[i] > lowest_since_entry + 2.5 * atr_6h[i]
                exit_condition = trailing_stop
            
            if exit_condition:
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals