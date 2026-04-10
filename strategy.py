#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + ATR(14) filter
# - Long when price breaks above Donchian high(20) AND volume > 1.5x 20-bar avg AND ATR(14) > ATR(50) (volatility expansion)
# - Short when price breaks below Donchian low(20) AND volume > 1.5x 20-bar avg AND ATR(14) > ATR(50)
# - Exit when price crosses Donchian midline (10-bar avg of high/low) or ATR(14) < ATR(50)*0.8 (volatility contraction)
# - Uses discrete position sizing (0.30) to control drawdown
# - Targets ~20-40 trades/year (80-160 total over 4 years) to avoid fee drag
# - Donchian channels provide clear structure, volume confirms participation, ATR filter ensures trades occur in expanding volatility regimes
# - Works in bull markets (breakouts continue) and bear markets (breakdowns continue) with volatility filter preventing false signals in ranging periods

name = "4h_donchian_breakout_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for additional filter if needed)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # ATR (14 and 50 for volatility regime filter)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Volume confirmation: > 1.5x 20-period average
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * volume_20_avg)
    
    # Optional 1d trend filter (only use if 1d data available)
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
        ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
        # 1d uptrend = close > EMA50, downtrend = close < EMA50
        trend_filter_long = close > ema50_1d_aligned
        trend_filter_short = close < ema50_1d_aligned
    else:
        trend_filter_long = np.ones(n, dtype=bool)
        trend_filter_short = np.ones(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after ATR50 warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr14[i]) or np.isnan(atr50[i]) or 
            np.isnan(volume_20_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: breakout above Donchian high + volume spike + volatility expansion + 1d uptrend filter
            if (close[i] > donchian_high[i] and 
                vol_spike[i] and 
                atr14[i] > atr50[i] and  # Volatility expansion
                trend_filter_long[i]):
                position = 1
                signals[i] = 0.30
            # Short: breakdown below Donchian low + volume spike + volatility expansion + 1d downtrend filter
            elif (close[i] < donchian_low[i] and 
                  vol_spike[i] and 
                  atr14[i] > atr50[i] and  # Volatility expansion
                  trend_filter_short[i]):
                position = -1
                signals[i] = -0.30
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price crosses Donchian midline (mean reversion signal)
            # 2. Volatility contraction (ATR14 < ATR50 * 0.8)
            if position == 1:
                if (close[i] < donchian_mid[i] or 
                    atr14[i] < atr50[i] * 0.8):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.30  # Hold long
            elif position == -1:
                if (close[i] > donchian_mid[i] or 
                    atr14[i] < atr50[i] * 0.8):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.30  # Hold short
    
    return signals