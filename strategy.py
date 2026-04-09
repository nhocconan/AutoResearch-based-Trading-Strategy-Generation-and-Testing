#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Williams %R with 12h regime filter
# - Elder Ray (Bull/Bear Power) measures buying/selling pressure relative to EMA
# - Williams %R identifies overbought/oversold conditions
# - 12h ADX regime filter: only trade when ADX > 25 (trending) for Elder Ray signals
# - In ranging markets (ADX <= 25), fade Williams %R extremes at 12h pivot levels
# - Position size: 0.25 to manage drawdown in volatile 6h timeframe
# - Target: 12-30 trades/year (50-120 total over 4 years)
# - Works in both bull/bear: trend following in trends, mean reversion in ranges

name = "6h_12h_elderray_williamsr_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h indicators
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h EMA(21) for Elder Ray
    ema_21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 12h True Range for ATR (used in ADX)
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_12h[0] = tr_12h[0]
    
    # 12h ATR(14) for ADX calculation
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # 12h ADX calculation
    # +DM and -DM
    up_move = high_12h - np.roll(high_12h, 1)
    down_move = np.roll(low_12h, 1) - low_12h
    up_move = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    down_move = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed +DM, -DM, ATR
    plus_dm = pd.Series(up_move).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm = pd.Series(down_move).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_ma = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm / np.where(atr_ma == 0, 1, atr_ma)
    minus_di = 100 * minus_dm / np.where(atr_ma == 0, 1, atr_ma)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1, (plus_di + minus_di))
    adx_12h = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # 12h Pivot Points (for mean reversion in ranging markets)
    pp_12h = (high_12h + low_12h + close_12h) / 3
    r1_12h = 2 * pp_12h - low_12h
    s1_12h = 2 * pp_12h - high_12h
    
    # Align 12h indicators to 6h
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    pp_12h_aligned = align_htf_to_ltf(prices, df_12h, pp_12h)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 6h Williams %R (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / np.where((highest_high_14 - lowest_low_14) == 0, 1, (highest_high_14 - lowest_low_14))
    
    # 6h Elder Ray (Bull/Bear Power)
    bull_power = high - ema_21_12h_aligned  # Using 12h EMA for multi-timeframe
    bear_power = low - ema_21_12h_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_12h_aligned[i]) or 
            np.isnan(williams_r[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(pp_12h_aligned[i]) or
            np.isnan(r1_12h_aligned[i]) or
            np.isnan(s1_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX > 25 = trending, ADX <= 25 = ranging
        is_trending = adx_12h_aligned[i] > 25
        
        if position == 0:  # Flat - look for entry
            if is_trending:
                # Trending market: Elder Ray signals
                if bull_power[i] > 0 and bear_power[i] < 0:  # Strong bullish
                    signals[i] = 0.25
                    position = 1
                elif bear_power[i] < 0 and bull_power[i] < 0:  # Strong bearish
                    signals[i] = -0.25
                    position = -1
            else:
                # Ranging market: Williams %R mean reversion at pivot levels
                if williams_r[i] <= -80 and low[i] <= s1_12h_aligned[i]:  # Oversold + at S1
                    signals[i] = 0.25
                    position = 1
                elif williams_r[i] >= -20 and high[i] >= r1_12h_aligned[i]:  # Overbought + at R1
                    signals[i] = -0.25
                    position = -1
        elif position == 1:  # Long position - look for exit
            if is_trending:
                # Exit on bearish Elder Ray divergence
                if bear_power[i] > 0:  # Bears taking over
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # Exit at pivot or opposite extreme
                if williams_r[i] >= -20 or high[i] >= r1_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:  # Short position - look for exit
            if is_trending:
                # Exit on bullish Elder Ray divergence
                if bull_power[i] < 0:  # Bulls taking over
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # Exit at pivot or opposite extreme
                if williams_r[i] <= -80 or low[i] <= s1_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals