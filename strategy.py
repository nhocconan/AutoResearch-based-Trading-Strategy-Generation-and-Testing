#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and 1d volatility regime
# - Primary signal: Price breaks above/below Camarilla H3/L3 levels on 1h (calculated from prior 4h bar)
# - Trend filter: 4h close > 4h EMA50 for longs, < EMA50 for shorts (avoid counter-trend)
# - Volatility filter: 1d ATR(14) < 0.04 * price (avoid extremely high volatility chop)
# - Session filter: Trade only 08:00-20:00 UTC (avoid low-liquidity Asian session)
# - Position size: 0.20 discrete level to minimize fee churn
# - Stoploss: 1.5x ATR(14) on 1h
# - Target: 15-37 trades/year (60-150 total over 4 years) per 1h strategy guidelines

name = "1h_4h_1d_camarilla_pivot_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Pre-compute 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr_1d1 = high_1d - low_1d
    tr_1d2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr_1d3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr_1d1, np.maximum(tr_1d2, tr_1d3))
    tr_1d[0] = tr_1d1[0]
    atr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    vol_filter = (atr_14 / close_1d) < 0.04  # ATR < 4% of price
    vol_filter_aligned = align_htf_to_ltf(prices, df_1d, vol_filter)
    
    # Pre-compute Camarilla levels from prior 4h bar (H3, L3, H4, L4)
    # Camarilla: H4 = close + 1.1*(high-low)/2, H3 = close + 1.1*(high-low)/4
    #            L3 = close - 1.1*(high-low)/4, L4 = close - 1.1*(high-low)/2
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for each 4h bar
    camarilla_h3 = close_4h + 1.1 * (high_4h - low_4h) / 4
    camarilla_l3 = close_4h - 1.1 * (high_4h - low_4h) / 4
    camarilla_h4 = close_4h + 1.1 * (high_4h - low_4h) / 2
    camarilla_l4 = close_4h - 1.1 * (high_4h - low_4h) / 2
    
    # Align Camarilla levels to 1h timeframe (use prior completed 4h bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l4)
    
    # Pre-compute 1h ATR(14) for stoploss
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    
    tr_1h1 = high_1h - low_1h
    tr_1h2 = np.abs(high_1h - np.roll(close_1h, 1))
    tr_1h3 = np.abs(low_1h - np.roll(close_1h, 1))
    tr_1h = np.maximum(tr_1h1, np.maximum(tr_1h2, tr_1h3))
    tr_1h[0] = tr_1h1[0]
    atr_14_1h = pd.Series(tr_1h).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute session filter (08:00-20:00 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_aligned[i]) or np.isnan(vol_filter_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(atr_14_1h[i])):
            signals[i] = 0.0
            continue
        
        # Check session filter
        if not session_filter[i]:
            if position != 0:
                # Exit at session close if still in position
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Camarilla mean reversion (below L3) OR stoploss hit
            if close_1h[i] < camarilla_l3_aligned[i] or close_1h[i] < entry_price - 1.5 * atr_14_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: Camarilla mean reversion (above H3) OR stoploss hit
            if close_1h[i] > camarilla_h3_aligned[i] or close_1h[i] > entry_price + 1.5 * atr_14_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for Camarilla breakouts with trend and volatility filters
            if vol_filter_aligned[i]:
                # Long: price breaks above Camarilla H3 with 4h uptrend
                if close_1h[i] > camarilla_h3_aligned[i] and close_1h[i] > ema_50_aligned[i]:
                    position = 1
                    entry_price = close_1h[i]
                    signals[i] = 0.20
                # Short: price breaks below Camarilla L3 with 4h downtrend
                elif close_1h[i] < camarilla_l3_aligned[i] and close_1h[i] < ema_50_aligned[i]:
                    position = -1
                    entry_price = close_1h[i]
                    signals[i] = -0.20
    
    return signals