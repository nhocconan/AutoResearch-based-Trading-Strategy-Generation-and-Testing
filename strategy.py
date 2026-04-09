#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d HTF regime filters with Donchian breakout entries
# Uses 4h trend (EMA21 > EMA50) and 1d chop regime (BB Width < 0.06) to confirm trending markets
# Enters long on 1h Donchian(20) breakout with volume spike, short on breakdown
# In ranging regimes (1d BB Width > 0.12), fades 1h Donchian extremes
# Session filter 08-20 UTC reduces noise. Position size 0.20 targets ~20-40 trades/year.
# Works in bull/bear: breakouts capture trends in trending regimes, mean reversion works in ranging markets

name = "1h_4h_1d_donchian_chop_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values if 'volume' in df_4h.columns else np.zeros_like(close_4h)
    
    # 4h EMA21 and EMA50 for trend filter
    close_s_4h = pd.Series(close_4h)
    ema21_4h = close_s_4h.ewm(span=21, min_periods=21, adjust=False).mean().values
    ema50_4h = close_s_4h.ewm(span=50, min_periods=50, adjust=False).mean().values
    uptrend_4h = ema21_4h > ema50_4h
    downtrend_4h = ema21_4h < ema50_4h
    
    # Align 4h indicators to 1h
    uptrend_4h_aligned = align_htf_to_ltf(prices, df_4h, uptrend_4h)
    downtrend_4h_aligned = align_htf_to_ltf(prices, df_4h, downtrend_4h)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d Bollinger Bands for chop regime
    close_s_1d = pd.Series(close_1d)
    basis_1d = close_s_1d.rolling(window=20, min_periods=20).mean().values
    dev_1d = close_s_1d.rolling(window=20, min_periods=20).std().values
    upper_bb_1d = basis_1d + 2.0 * dev_1d
    lower_bb_1d = basis_1d - 2.0 * dev_1d
    bb_width_1d = (upper_bb_1d - lower_bb_1d) / basis_1d
    bb_width_1d = np.where(basis_1d != 0, bb_width_1d, 0)
    
    # Align 1d BB width to 1h
    bb_width_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_width_1d)
    
    # 1h Donchian channels (20-period)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    
    # 1h volume spike confirmation (20-period avg)
    vol_s = pd.Series(volume)
    vol_ma_20 = vol_s.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (not in_session[i] or 
            np.isnan(uptrend_4h_aligned[i]) or np.isnan(downtrend_4h_aligned[i]) or
            np.isnan(bb_width_1d_aligned[i]) or np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Regime filters
        trending_regime = uptrend_4h_aligned[i] or downtrend_4h_aligned[i]  # 4h trend present
        chop_regime = bb_width_1d_aligned[i] > 0.12  # 1d high BB width = ranging
        
        if position == 1:  # Long position
            if trending_regime and not chop_regime:
                # Exit long if price falls below Donchian low or volume drops
                if close[i] < donchian_low[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            else:  # Chop regime or no trend - mean reversion
                # Exit long if price moves back above Donchian mid
                donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
                if close[i] > donchian_mid:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
                    
        elif position == -1:  # Short position
            if trending_regime and not chop_regime:
                # Exit short if price rises above Donchian high
                if close[i] > donchian_high[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
            else:  # Chop regime or no trend - mean reversion
                # Exit short if price moves back below Donchian mid
                donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
                if close[i] < donchian_mid:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
        else:  # Flat
            if trending_regime and not chop_regime:
                # Breakout strategy in trending market (no chop)
                if close[i] > donchian_high[i] and volume_spike[i]:
                    position = 1
                    signals[i] = 0.20
                elif close[i] < donchian_low[i] and volume_spike[i]:
                    position = -1
                    signals[i] = -0.20
            else:  # Chop regime - mean reversion at extremes
                if close[i] < donchian_low[i]:
                    position = 1
                    signals[i] = 0.20
                elif close[i] > donchian_high[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals