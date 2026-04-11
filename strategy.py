#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Long: price breaks above Camarilla H3 level, volume > 1.5x 20-period avg, 1w close > 1w EMA(20) (bullish trend)
# - Short: price breaks below Camarilla L3 level, volume > 1.5x 20-period avg, 1w close < 1w EMA(20) (bearish trend)
# - Exit: price returns to Camarilla pivot point (mid-level)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 15-25 trades/year (60-100 total over 4 years) to stay within fee drag limits
# - Camarilla pivots work well in ranging markets; 1w EMA filter ensures we only trade with the higher timeframe trend
# - Volume confirmation reduces false breakouts

name = "1d_1w_camarilla_pivot_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Pre-compute 1d Camarilla pivot levels (based on previous day's OHLC)
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # We'll use H3 for long entry and L3 for short entry
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point (PP) = (H + L + C) / 3
    pp_1d = (high_1d + low_1d + close_1d) / 3
    # Range = H - L
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    h3_1d = pp_1d + range_1d * 1.1 / 4
    l3_1d = pp_1d - range_1d * 1.1 / 4
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return signals
    
    # Pre-compute 1w EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(pp_1d_aligned[i]) or np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Camarilla levels
        h3 = h3_1d_aligned[i]
        l3 = l3_1d_aligned[i]
        pp = pp_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # 1w trend filter: close > EMA(20) for bullish, close < EMA(20) for bearish
        ema_1w = ema_20_1w_aligned[i]
        trend_bullish = close_price > ema_1w
        trend_bearish = close_price < ema_1w
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price above Camarilla H3, volume confirmation, bullish 1w trend
        if close_price > h3 and vol_confirm and trend_bullish:
            enter_long = True
        
        # Short breakout: price below Camarilla L3, volume confirmation, bearish 1w trend
        if close_price < l3 and vol_confirm and trend_bearish:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to pivot point
            exit_long = close_price <= pp
        elif position == -1:
            # Exit short if price returns to pivot point
            exit_short = close_price >= pp
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals