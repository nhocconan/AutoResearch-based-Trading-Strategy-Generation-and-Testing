#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud + Volume Spike + 1d ADX Trend Filter
# Uses Ichimoku (TK cross + cloud) for momentum signals on 6h, confirmed by 1d ADX > 25 for trend strength.
# Volume spike (20-period volume > 1.5 * 50-period avg volume) adds conviction.
# Long when: TK cross bullish, price above cloud, ADX > 25, volume spike.
# Short when: TK cross bearish, price below cloud, ADX > 25, volume spike.
# Ichimoku captures momentum and support/resistance; ADX filters for trending markets only; volume confirms conviction.
# Works in bull (trend continuation) and bear (trend continuation) by aligning with higher timeframe trend.
# Discrete sizing 0.25 balances return and drawdown. Target: 15-30 trades/year.

name = "6h_Ichimoku_ADX_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(values, period):
        """Wilder's smoothing (equivalent to EMA with alpha=1/period)"""
        if len(values) < period:
            return np.full_like(values, np.nan)
        result = np.full_like(values, np.nan)
        result[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] - (result[i-1] / period) + values[i]
            else:
                result[i] = np.nan
        return result
    
    period_adx = 14
    tr_smooth = wilders_smoothing(tr, period_adx)
    plus_dm_smooth = wilders_smoothing(plus_dm, period_adx)
    minus_dm_smooth = wilders_smoothing(minus_dm, period_adx)
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = wilders_smoothing(dx, period_adx)
    
    # Align 1d ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Ichimoku on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2.0)
    # Shift will be handled by align_htf_to_ltf logic? No, we need to shift the values themselves for cloud
    # Actually, for Ichimoku cloud, Senkou Span A/B are plotted 26 periods ahead.
    # But for signal generation at time t, we use Senkou A/B from t-26 (already published)
    # So we calculate Senkou A/B then shift forward by 26 for plotting, but for our logic we use unshifted?
    # Standard use: price vs cloud where cloud is Senkou A/B shifted 26 ahead.
    # So at time t, cloud's leading edge is Senkou A/B from t-26.
    # Therefore, we calculate Senkou A/B then shift forward by 26 to get the cloud position.
    senkou_a_shifted = np.roll(senkou_a, -period_kijun)  # shift left by 26 for cloud
    senkou_a_shifted[-period_kijun:] = np.nan  # pad end with nan
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_senkou_b + min_low_senkou_b) / 2.0)
    senkou_b_shifted = np.roll(senkou_b, -period_kijun)  # shift left by 26 for cloud
    senkou_b_shifted[-period_kijun:] = np.nan
    
    # Chikou Span (Lagging Span): close shifted -22 periods (not used in signal)
    # We don't use Chikou for simplicity
    
    # Volume Spike: 20-period volume > 1.5 * 50-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = vol_ma_20 > (1.5 * vol_ma_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, period_kijun, period_senkou_b) + period_kijun  # ensure Ichimoku and ADX ready
    
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
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a_shifted[i]) or np.isnan(senkou_b_shifted[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_tenkan = tenkan[i]
        curr_kijun = kijun[i]
        curr_senkou_a = senkou_a_shifted[i]
        curr_senkou_b = senkou_b_shifted[i]
        curr_adx = adx_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        # Ichimoku conditions
        # Bullish TK cross: Tenkan > Kijun
        tk_bullish = curr_tenkan > curr_kijun
        # Bearish TK cross: Tenkan < Kijun
        tk_bearish = curr_tenkan < curr_kijun
        # Price above cloud: price > Senkou A and price > Senkou B
        cloud_top = max(curr_senkou_a, curr_senkou_b)
        cloud_bottom = min(curr_senkou_a, curr_senkou_b)
        price_above_cloud = curr_close > cloud_top
        price_below_cloud = curr_close < cloud_bottom
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Bullish TK cross, price above cloud, ADX > 25, volume spike
            if (tk_bullish and 
                price_above_cloud and 
                curr_adx > 25 and 
                curr_volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Bearish TK cross, price below cloud, ADX > 25, volume spike
            elif (tk_bearish and 
                  price_below_cloud and 
                  curr_adx > 25 and 
                  curr_volume_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bearish TK cross OR price breaks below cloud OR ADX < 20 (trend weakening)
            if (tk_bearish or 
                not price_above_cloud or 
                curr_adx < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bullish TK cross OR price breaks above cloud OR ADX < 20 (trend weakening)
            if (tk_bullish or 
                not price_below_cloud or 
                curr_adx < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals